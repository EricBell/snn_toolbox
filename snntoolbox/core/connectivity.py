"""Pure-Python library for SNN connectivity calculations.

Every function in this module operates on plain numpy arrays and
:class:`~snntoolbox.core.ir.IRModel` / :class:`~snntoolbox.core.ir.IRLayer`
objects.  **No ML framework imports** (TensorFlow, PyTorch, etc.) are used,
so the entire module can be unit-tested with nothing heavier than numpy.

The logic here is the framework-agnostic counterpart of the Keras-layer
bookkeeping that used to live in ``snntoolbox.parsing.utils`` and
``snntoolbox.simulation.utils``.  It answers questions the spiking backends
ask while allocating neurons and counting synaptic operations:

* How many inputs does a neuron in *this* layer receive?           (fan-in)
* How many outputs does a neuron in *this* layer drive?            (fan-out)
* How many neurons does this layer contain?                        (num_neurons)
* Does this layer get converted into spiking neurons at all?       (is_spiking)
* What are the total neuron / bias / fan-in / synapse counts for
  an ``IRModel``?                                                  (ConnectivityStats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
)

# Layer types that become spiking neurons at inference time. Concatenate is
# a graph-structure operator and gets merged into upstream spiking layers.
_SPIKING_LAYER_TYPES: frozenset[LayerType] = frozenset({
    LayerType.DENSE,
    LayerType.CONV1D,
    LayerType.CONV2D,
    LayerType.CONV2D_TRANSPOSE,
    LayerType.DEPTHWISE_CONV2D,
    LayerType.AVERAGE_POOLING_2D,
    LayerType.MAX_POOLING_2D,
    LayerType.SPARSE,
    LayerType.SPARSE_CONV2D,
    LayerType.SPARSE_DEPTHWISE_CONV2D,
})

_CONV_LAYER_TYPES: frozenset[LayerType] = frozenset({
    LayerType.CONV1D,
    LayerType.CONV2D,
    LayerType.CONV2D_TRANSPOSE,
    LayerType.DEPTHWISE_CONV2D,
    LayerType.SPARSE_CONV2D,
    LayerType.SPARSE_DEPTHWISE_CONV2D,
})

_DEPTHWISE_LAYER_TYPES: frozenset[LayerType] = frozenset({
    LayerType.DEPTHWISE_CONV2D,
    LayerType.SPARSE_DEPTHWISE_CONV2D,
})

_POOL_LAYER_TYPES: frozenset[LayerType] = frozenset({
    LayerType.AVERAGE_POOLING_2D,
    LayerType.MAX_POOLING_2D,
})


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def is_spiking(layer: IRLayer) -> bool:
    """Return ``True`` if *layer* converts to spiking neurons."""
    return layer.layer_type in _SPIKING_LAYER_TYPES


def is_conv(layer: IRLayer) -> bool:
    """Return ``True`` if *layer* is any flavour of convolution."""
    return layer.layer_type in _CONV_LAYER_TYPES


def is_depthwise_conv(layer: IRLayer) -> bool:
    """Return ``True`` if *layer* is a depthwise convolution."""
    return layer.layer_type in _DEPTHWISE_LAYER_TYPES


def is_pool(layer: IRLayer) -> bool:
    """Return ``True`` if *layer* is a pooling operation."""
    return layer.layer_type in _POOL_LAYER_TYPES


def has_stride_unity(layer: IRLayer) -> bool:
    """Return ``True`` if all stride dimensions of *layer* equal 1."""
    strides = layer.strides or (1,)
    return all(s == 1 for s in strides)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _channels_first(layer: IRLayer, model: Optional[IRModel] = None) -> bool:
    """Determine the channel axis convention for *layer*."""
    fmt = layer.data_format
    if model is not None and fmt is None:
        fmt = model.data_format
    return fmt == DataFormat.CHANNELS_FIRST


def _shape_without_batch(
    layer: IRLayer, model: Optional[IRModel] = None,
) -> tuple[int, ...]:
    """Return *layer*'s spatial output shape with the batch dim stripped.

    The IR's ``output_shape`` is always ``(batch_or_None, *spatial_dims)``,
    so we always drop the first element regardless of whether the batch
    dimension has been concretized (e.g. after ``model.build``).
    """
    shape = layer.output_shape
    if shape:
        shape = shape[1:]
    if not shape and model is not None and layer.layer_type == LayerType.INPUT:
        shape = tuple(model.input_shape)
    return tuple(shape)


def num_neurons(layer: IRLayer, model: Optional[IRModel] = None) -> int:
    """Total neuron count for *layer* (product of spatial output dims)."""
    shape = _shape_without_batch(layer, model)
    if not shape:
        return 0
    return int(np.prod(shape))


# ---------------------------------------------------------------------------
# Graph navigation
# ---------------------------------------------------------------------------


def get_spiking_outbound_layers(
    layer: IRLayer, model: IRModel,
) -> list[IRLayer]:
    """Walk outbound edges until every branch hits a spiking layer.

    This is the IR counterpart of
    ``snntoolbox.simulation.utils.get_spiking_outbound_layers``.  The
    behaviour matches the original: non-spiking intermediates (Flatten,
    Reshape, Concatenate, ZeroPadding, …) are transparently skipped so
    that the caller only sees the next *spiking* layer along each branch.
    """
    name_to_outbound: dict[str, list[IRLayer]] = {lyr.name: [] for lyr in model.layers}
    for lyr in model.layers:
        for inbound_name in lyr.inbound:
            if inbound_name in name_to_outbound:
                name_to_outbound[inbound_name].append(lyr)

    def _walk(start: IRLayer) -> list[IRLayer]:
        outbound = name_to_outbound.get(start.name, [])
        if not outbound:
            return []
        if len(outbound) == 1:
            nxt = outbound[0]
            if is_spiking(nxt):
                return [nxt]
            return _walk(nxt)
        result: list[IRLayer] = []
        for nxt in outbound:
            if is_spiking(nxt):
                result.append(nxt)
            else:
                result.extend(_walk(nxt))
        return result

    return _walk(layer)


# ---------------------------------------------------------------------------
# Fan-in
# ---------------------------------------------------------------------------


def get_fanin(layer: IRLayer, model: Optional[IRModel] = None) -> int:
    """Return the fan-in of a neuron in *layer*.

    * Convolution: ``prod(kernel_size) * in_channels``
    * Dense: ``in_features``
    * Pool / everything else: ``0``
    """
    if is_conv(layer):
        ax = 1 if _channels_first(layer, model) else -1
        # input_shape stored in extra_config; fall back to inbound layer's
        # output_shape if absent.
        in_shape = layer.input_shape
        if in_shape is None and model is not None and layer.inbound:
            inbound = model.get_layer(layer.inbound[0])
            if inbound is not None:
                in_shape = inbound.output_shape
        if in_shape is None:
            return 0
        channels = in_shape[ax]
        if channels is None:
            return 0
        kernel = layer.kernel_size or (1,)
        return int(np.prod(kernel)) * int(channels)

    if layer.layer_type in (LayerType.DENSE, LayerType.SPARSE):
        in_shape = layer.input_shape
        if in_shape is None and model is not None and layer.inbound:
            inbound = model.get_layer(layer.inbound[0])
            if inbound is not None:
                in_shape = inbound.output_shape
        if in_shape is None:
            return 0
        return int(in_shape[1])

    return 0


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def get_fanout(layer: IRLayer, model: IRModel):
    """Return the fan-out of a neuron in *layer*.

    If any downstream convolution uses ``stride > 1`` the fan-out varies
    between neurons of *layer*; in that case an ndarray with the same
    shape as the layer (batch dim stripped) is returned.  Otherwise a
    plain ``int`` is returned.
    """
    next_layers = get_spiking_outbound_layers(layer, model)
    fanout: float | np.ndarray = 0
    for nxt in next_layers:
        if is_conv(nxt) and not has_stride_unity(nxt):
            shape = _shape_without_batch(layer, model)
            fanout = np.zeros(shape)
            break

    for nxt in next_layers:
        if nxt.layer_type in (LayerType.DENSE, LayerType.SPARSE):
            fanout = fanout + (nxt.output_shape[-1] or 0)
        elif is_pool(nxt):
            fanout = fanout + 1
        elif is_depthwise_conv(nxt):
            if has_stride_unity(nxt):
                fanout = fanout + int(np.prod(nxt.kernel_size or (1,)))
            else:
                fanout = fanout + get_fanout_array(
                    layer, nxt, model, is_depthwise=True,
                )
        elif is_conv(nxt):
            if has_stride_unity(nxt):
                kprod = int(np.prod(nxt.kernel_size or (1,)))
                filters = nxt.filters if nxt.filters is not None else 0
                fanout = fanout + kprod * filters
            else:
                fanout = fanout + get_fanout_array(layer, nxt, model)

    return fanout


def get_fanout_array(
    layer_pre: IRLayer,
    layer_post: IRLayer,
    model: Optional[IRModel] = None,
    is_depthwise: bool = False,
) -> np.ndarray:
    """Per-neuron fan-out array for a stride>1 conv going into *layer_post*.

    Returns an ndarray shaped like *layer_pre*'s spatial output (batch dim
    stripped) where each entry gives the number of post-synaptic neurons
    driven by the pre-synaptic neuron at that location.
    """
    shape = _shape_without_batch(layer_pre, model)
    ndim = len(shape)

    if ndim == 3:
        return _fanout_array_2d(layer_pre, layer_post, model, is_depthwise)
    if ndim == 2:
        return _fanout_array_1d(layer_pre, layer_post, model, is_depthwise)
    raise NotImplementedError(
        f'get_fanout_array is only defined for 1D/2D feature maps '
        f'(pre-layer spatial rank {ndim}).'
    )


def _fanout_array_1d(
    layer_pre: IRLayer,
    layer_post: IRLayer,
    model: Optional[IRModel],
    is_depthwise: bool,
) -> np.ndarray:
    ax = 1 if _channels_first(layer_pre, model) else 0

    post_shape = layer_post.output_shape
    ny = post_shape[2] if ax else post_shape[1]
    nz = post_shape[1] if ax else post_shape[-1]
    ky = (layer_post.kernel_size or (1,))[0]
    py = int((ky - 1) / 2) if layer_post.padding == 'same' else 0
    sy = (layer_post.strides or (1,))[0]

    shape = _shape_without_batch(layer_pre, model)
    fanout = np.zeros(shape)

    for y_pre in range(fanout.shape[0 + ax]):
        y_post = [int((y_pre + py) / sy)]
        wy = (y_pre + py) % sy
        i = 1
        while wy + i * sy < ky:
            y = y_post[0] - i
            if 0 <= y < ny:
                y_post.append(y)
            i += 1

        if ax:
            fanout[:, y_pre] = len(y_post)
        else:
            fanout[y_pre, :] = len(y_post)

    if not is_depthwise:
        fanout *= nz

    return fanout


def _fanout_array_2d(
    layer_pre: IRLayer,
    layer_post: IRLayer,
    model: Optional[IRModel],
    is_depthwise: bool,
) -> np.ndarray:
    ax = 1 if _channels_first(layer_pre, model) else 0

    post_shape = layer_post.output_shape
    nx = post_shape[3] if ax else post_shape[2]
    ny = post_shape[2] if ax else post_shape[1]
    nz = post_shape[1] if ax else post_shape[-1]
    kx, ky = (layer_post.kernel_size or (1, 1))
    px = int((kx - 1) / 2) if layer_post.padding == 'same' else 0
    py = int((ky - 1) / 2) if layer_post.padding == 'same' else 0
    strides = layer_post.strides or (1, 1)
    sx = strides[1]
    sy = strides[0]

    shape = _shape_without_batch(layer_pre, model)
    fanout = np.zeros(shape)

    for y_pre in range(fanout.shape[0 + ax]):
        y_post = [int((y_pre + py) / sy)]
        wy = (y_pre + py) % sy
        i = 1
        while wy + i * sy < ky:
            y = y_post[0] - i
            if 0 <= y < ny:
                y_post.append(y)
            i += 1
        for x_pre in range(fanout.shape[1 + ax]):
            x_post = [int((x_pre + px) / sx)]
            wx = (x_pre + px) % sx
            i = 1
            while wx + i * sx < kx:
                x = x_post[0] - i
                if 0 <= x < nx:
                    x_post.append(x)
                i += 1

            if ax:
                fanout[:, y_pre, x_pre] = len(x_post) * len(y_post)
            else:
                fanout[y_pre, x_pre, :] = len(x_post) * len(y_post)

    if not is_depthwise:
        fanout *= nz

    return fanout


# ---------------------------------------------------------------------------
# Aggregate connectivity stats for a whole model
# ---------------------------------------------------------------------------


@dataclass
class ConnectivityStats:
    """Aggregate connectivity statistics for an :class:`IRModel`.

    Attributes
    ----------
    num_neurons
        Neurons per layer, starting with the input layer.
    num_neurons_with_bias
        Per-layer count of neurons whose preceding weight layer has a
        non-zero bias (0 for layers without weights).
    fanin
        Per-neuron fan-in per layer.  Input layer is 0.
    fanout
        Per-neuron fan-out per layer.  Either ``int`` or ``ndarray`` (for
        conv layers with stride > 1).
    num_synapses
        Total synapse count across the whole network.
    """

    num_neurons: list[int]
    num_neurons_with_bias: list[int]
    fanin: list[int]
    fanout: list
    num_synapses: int


def compute_connectivity(
    model: IRModel,
    bias_predicate=None,
) -> ConnectivityStats:
    """Compute neuron / synapse counts for every layer in *model*.

    Parameters
    ----------
    model
        The framework-agnostic model.
    bias_predicate
        Optional callable ``layer -> bool`` that decides whether a layer's
        bias counts as "used".  Defaults to "has a non-zero bias vector".
    """
    if bias_predicate is None:
        bias_predicate = _default_bias_predicate

    fanin: list[int] = [0]
    fanout: list = [get_fanout(model.input_layer, model)]
    num_neurons_list: list[int] = [num_neurons(model.input_layer, model)]
    num_neurons_with_bias: list[int] = [0]

    for layer in model.layers:
        if layer.layer_type == LayerType.INPUT:
            continue
        if not is_spiking(layer):
            continue

        fanin.append(get_fanin(layer, model))
        fanout.append(get_fanout(layer, model))
        n = num_neurons(layer, model)
        num_neurons_list.append(n)
        num_neurons_with_bias.append(n if bias_predicate(layer) else 0)

    total_synapses = 0
    for i, fo in enumerate(fanout):
        if np.isscalar(fo):
            total_synapses += num_neurons_list[i] * int(fo)
        else:
            total_synapses += int(np.sum(fo))

    return ConnectivityStats(
        num_neurons=num_neurons_list,
        num_neurons_with_bias=num_neurons_with_bias,
        fanin=fanin,
        fanout=fanout,
        num_synapses=int(total_synapses),
    )


def _default_bias_predicate(layer: IRLayer) -> bool:
    """Return ``True`` if *layer* has a weight tensor with any non-zero bias."""
    if layer.weights is None or layer.weights.bias is None:
        return False
    return bool(np.any(layer.weights.bias))


# ---------------------------------------------------------------------------
# ANN operation count (moved here because it's a pure function of the
# connectivity vectors, not of any framework object)
# ---------------------------------------------------------------------------


def compute_ann_ops(
    num_neurons_list: Sequence[int],
    num_neurons_with_bias: Sequence[int],
    fanin: Sequence[int],
) -> int:
    """Number of multiply-adds performed by an ANN in one forward pass."""
    return int(2 * np.dot(num_neurons_list, fanin) + np.sum(num_neurons_with_bias))
