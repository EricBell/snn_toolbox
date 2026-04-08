"""Pure-Python library for spiking neuron parameter calculations.

Every function in this module operates on plain numpy arrays and
:class:`~snntoolbox.core.ir.IRModel` / :class:`~snntoolbox.core.ir.IRLayer`
objects.  **No ML framework imports** (TensorFlow, PyTorch, etc.) are used,
so the entire module can be unit-tested with nothing heavier than numpy.

Functions fall into three categories:

1. **Weight transforms** — batch-norm absorption, binarization, precision
   reduction.
2. **Scale-factor computation** — percentile-based activation statistics.
3. **IR-level normalization** — :func:`normalize_weights` and
   :func:`compute_scale_factors` operate on :class:`IRModel` and return a
   new, normalized model.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from snntoolbox.core.ir import IRLayer, IRModel, LayerWeights

logger = logging.getLogger('snntoolbox.core.spiking_params')

# ---------------------------------------------------------------------------
# Weight transforms
# ---------------------------------------------------------------------------


def absorb_bn_parameters(
    weight: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var_eps_sqrt_inv: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    axis: int,
    image_data_format: str,
    is_depthwise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse batch-normalization parameters into the preceding layer.

    This is a pure numpy operation: it reshapes BN statistics to broadcast
    against the weight tensor, then computes the fused weight and bias.

    Parameters
    ----------
    weight : np.ndarray
        Weight tensor of the preceding layer.
    bias : np.ndarray
        Bias vector of the preceding layer.
    mean, var_eps_sqrt_inv, gamma, beta : np.ndarray
        Batch-normalization parameters.  ``var_eps_sqrt_inv`` is
        ``1 / sqrt(variance + epsilon)``.
    axis : int
        The channel axis in the *layer* output (not the kernel).
    image_data_format : str
        ``'channels_first'`` or ``'channels_last'``.
    is_depthwise : bool
        Whether the preceding layer is a depthwise convolution.

    Returns
    -------
    weight_bn, bias_bn : tuple[np.ndarray, np.ndarray]
    """
    axis = weight.ndim + axis if axis < 0 else axis

    # Map BN axis from layer-output space to kernel-dimension space.
    if weight.ndim == 4:  # Conv2D
        channel_axis = 2 if is_depthwise else 3
        if image_data_format == 'channels_first':
            layer2kernel = [None, channel_axis, 0, 1]
        else:
            layer2kernel = [None, 0, 1, channel_axis]
        axis = layer2kernel[axis]
    elif weight.ndim == 3:  # Conv1D
        channel_axis = 2
        if image_data_format == 'channels_first':
            layer2kernel = [None, channel_axis, 0]
        else:
            layer2kernel = [None, 0, channel_axis]
        axis = layer2kernel[axis]

    broadcast_shape = [1] * weight.ndim
    broadcast_shape[axis] = weight.shape[axis]

    var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
    gamma = np.reshape(gamma, broadcast_shape)
    beta = np.reshape(beta, broadcast_shape)
    bias = np.reshape(bias, broadcast_shape)
    mean = np.reshape(mean, broadcast_shape)

    bias_bn = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv)
    weight_bn = weight * gamma * var_eps_sqrt_inv

    return weight_bn, bias_bn


def binarize(
    w: np.ndarray,
    h: float = 1.0,
    deterministic: bool = True,
) -> np.ndarray:
    """Binarize weights to ``+/-h``.

    Parameters
    ----------
    w : np.ndarray
        Weight array.
    h : float
        Target magnitude (values become ``+h`` or ``-h``).
    deterministic : bool
        If ``True``, use deterministic rounding; otherwise stochastic.

    Returns
    -------
    np.ndarray
        Binarized weights (float32).
    """
    wb = np.clip((np.add(np.true_divide(w, h), 1.0)) / 2.0, 0, 1)
    wb = np.round(wb) if deterministic else np.random.binomial(1, wb)
    wb[wb != 0] = h
    wb[wb == 0] = -h
    return np.asarray(wb, np.float32)


def reduce_precision(x: np.ndarray, m: int, f: int) -> np.ndarray:
    """Reduce precision to fixed-point format ``Qm.f``.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    m : int
        Number of integer bits.
    f : int
        Number of fractional bits.

    Returns
    -------
    np.ndarray
        Data with reduced precision.
    """
    n = 2 << (f - 1)
    maxval = (2 << (m - 1)) - 1.0 / n
    return np.clip(np.true_divide(np.round(x * n), n), -maxval, maxval)


def modify_parameter_precision(
    weights: np.ndarray,
    biases: np.ndarray,
    config: Any,
    attributes: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally binarize or quantize layer parameters.

    Parameters
    ----------
    weights, biases : np.ndarray
        Layer parameters.
    config
        Toolbox configuration (must support ``getboolean``).
    attributes : dict
        Layer attributes dict; may contain ``'Qm.f'`` and
        ``'quantize_bias'`` keys.

    Returns
    -------
    weights, biases : tuple[np.ndarray, np.ndarray]
    """
    if config.getboolean('cell', 'binarize_weights'):
        logger.info('Binarizing weights.')
        weights = binarize(weights)
    elif config.getboolean('cell', 'quantize_weights'):
        assert 'Qm.f' in attributes, (
            "quantize_weights is True but layer has no 'Qm.f' attribute."
        )
        m, f = attributes.get('Qm.f')
        logger.info('Quantizing weights to Q%d.%d.', m, f)
        weights = reduce_precision(weights, m, f)
        if attributes.get('quantize_bias', False):
            biases = reduce_precision(biases, m, f)

    attributes.pop('quantize_bias', None)
    attributes.pop('Qm.f', None)

    return weights, biases


# ---------------------------------------------------------------------------
# Scale-factor computation
# ---------------------------------------------------------------------------


def get_scale_fac(activations: np.ndarray, percentile: float) -> float:
    """Activation value at the given percentile.

    Parameters
    ----------
    activations : np.ndarray
        Flattened non-zero activations of a layer.
    percentile : float
        Percentile (0–100).

    Returns
    -------
    float
        Scale factor.  Returns 1.0 if *activations* is empty.
    """
    return float(
        np.percentile(activations, percentile) if activations.size else 1.0
    )


def get_percentile(config: Any, layer_idx: Optional[int] = None) -> float:
    """Retrieve the normalization percentile, optionally with a schedule.

    Parameters
    ----------
    config
        Toolbox configuration.
    layer_idx : int, optional
        Layer index (required when a normalization schedule is active).

    Returns
    -------
    float
    """
    perc = config.getfloat('normalization', 'percentile')
    if config.getboolean('normalization', 'normalization_schedule'):
        assert layer_idx is not None and layer_idx >= 0, \
            'Layer index needed for normalization schedule.'
        perc = apply_normalization_schedule(perc, layer_idx)
    return perc


def apply_normalization_schedule(perc: float, layer_idx: int) -> float:
    """Decrease percentile for deeper layers to maintain spike rates.

    Parameters
    ----------
    perc : float
        Base percentile.
    layer_idx : int
        Zero-based layer index.

    Returns
    -------
    float
        Adjusted percentile.
    """
    return perc - layer_idx * 0.02


# ---------------------------------------------------------------------------
# IR-level normalization
# ---------------------------------------------------------------------------


def compute_scale_factors(
    ir_model: IRModel,
    layer_activations: dict[str, np.ndarray],
    config: Any,
) -> OrderedDict[str, float]:
    """Compute per-layer scale factors from pre-computed activations.

    This is the framework-agnostic replacement for the Keras-specific
    activation-computation loop in ``conversion/utils.py``.  Activations
    must be pre-computed by the caller using whatever framework produced
    the original model.

    Parameters
    ----------
    ir_model : IRModel
        The parsed model.
    layer_activations : dict[str, np.ndarray]
        Mapping of layer name → raw activation array.  Only layers with
        weights need entries.
    config
        Toolbox configuration.

    Returns
    -------
    OrderedDict[str, float]
        Scale factor for each layer (keyed by layer name).  The input
        layer is assigned a scale factor of 1.0.
    """
    scale_facs: OrderedDict[str, float] = OrderedDict()
    scale_facs[ir_model.input_layer.name] = 1.0

    idx = 0
    for layer in ir_model.layers:
        if not layer.has_weights:
            continue

        activations = layer_activations.get(layer.name)
        if activations is None:
            logger.warning(
                'No activations for layer %s; using scale_fac=1.0.',
                layer.name,
            )
            scale_facs[layer.name] = 1.0
            idx += 1
            continue

        nonzero = activations[np.nonzero(activations)]
        perc = get_percentile(config, idx)
        sf = get_scale_fac(nonzero, perc)
        scale_facs[layer.name] = sf
        logger.info('Layer %s: scale_fac=%.4f', layer.name, sf)
        idx += 1

    return scale_facs


def normalize_weights(
    ir_model: IRModel,
    scale_facs: dict[str, float],
) -> IRModel:
    """Normalize model weights using pre-computed scale factors.

    For each layer with weights, scales the kernel by
    ``scale_facs[inbound] / scale_facs[layer]`` and the bias by
    ``1 / scale_facs[layer]``.

    Returns a **new** :class:`IRModel` — the original is not mutated.

    Parameters
    ----------
    ir_model : IRModel
        The model whose weights to normalize.
    scale_facs : dict[str, float]
        Per-layer scale factors (from :func:`compute_scale_factors`).

    Returns
    -------
    IRModel
        A new model with normalized weights.
    """
    new_layers: list[IRLayer] = []

    for layer in ir_model.layers:
        if not layer.has_weights:
            new_layers.append(layer)
            continue

        w = layer.weights
        kernel = w.kernel.copy()
        bias = w.bias.copy()

        # Softmax layers use scale_fac=1 to avoid huge amplification
        if layer.activation == 'softmax':
            scale_fac = 1.0
            logger.debug('Using scale_fac=1.0 for softmax layer %s.',
                         layer.name)
        else:
            scale_fac = scale_facs.get(layer.name, 1.0)

        if scale_fac == 0:
            logger.warning('Scale factor is 0 for %s; skipping.', layer.name)
            new_layers.append(layer)
            continue

        # Resolve inbound layers that have weights
        inbound_with_weights = _get_inbound_with_weights(ir_model, layer)

        if len(inbound_with_weights) == 0:
            # First layer after input
            input_sf = scale_facs.get(ir_model.input_layer.name, 1.0)
            kernel = kernel * input_sf / scale_fac
            bias = bias / scale_fac
        elif len(inbound_with_weights) == 1:
            inb_sf = scale_facs.get(inbound_with_weights[0].name, 1.0)
            kernel = kernel * inb_sf / scale_fac
            bias = bias / scale_fac
        else:
            # Multi-input (e.g. after Concatenate): scale per input channel
            bias = bias / scale_fac
            if kernel.ndim == 4:
                offset = 0
                for inb in inbound_with_weights:
                    inb_sf = scale_facs.get(inb.name, 1.0)
                    f_out = inb.filters or inb.output_shape[-1]
                    kernel[:, :, offset:offset + f_out, :] *= (
                        inb_sf / scale_fac
                    )
                    offset += f_out
            else:
                # Flat concat — apply uniform scaling as fallback
                avg_sf = np.mean([
                    scale_facs.get(inb.name, 1.0)
                    for inb in inbound_with_weights
                ])
                kernel = kernel * avg_sf / scale_fac

        new_weights = LayerWeights(
            kernel=kernel,
            bias=bias,
            mask=w.mask,
        )
        new_layers.append(replace(layer, weights=new_weights))

    return IRModel(
        layers=new_layers,
        input_shape=ir_model.input_shape,
        data_format=ir_model.data_format,
        name=ir_model.name,
        original_framework=ir_model.original_framework,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_inbound_with_weights(
    ir_model: IRModel,
    layer: IRLayer,
) -> list[IRLayer]:
    """Walk inbound edges until layers with weights are found."""
    result: list[IRLayer] = []
    for name in layer.inbound:
        inb = ir_model.get_layer(name)
        if inb is None:
            continue
        if inb.has_weights:
            result.append(inb)
        else:
            # Recurse through weightless layers (Flatten, Concatenate, …)
            result.extend(_get_inbound_with_weights(ir_model, inb))
    return result
