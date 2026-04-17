"""Adapters bridging the legacy Keras-based interfaces to the new IR.

Two directions of adaptation:

**Legacy → IR** (so new backends can consume old parsers' output):

* :func:`keras_model_to_ir` — converts a ``keras.models.Model``
  (produced by existing parsers) into an :class:`IRModel`.
* :func:`layer_list_to_ir` — converts the internal ``_layer_list``
  (list of dicts) directly to IR, bypassing Keras model construction.

**IR → Legacy** (so old backends can consume IR without modification):

* :class:`IRLayerFacade` — wraps an :class:`IRLayer` to expose a
  Keras-layer-like API (``get_weights()``, ``.output_shape``, etc.).
* :class:`IRModelFacade` — wraps an :class:`IRModel` so that
  ``AbstractSNN.build(parsed_model)`` works transparently.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
    LAYER_TYPE_FROM_STRING,
    LAYER_TYPE_TO_STRING,
)


# -----------------------------------------------------------------------
# Legacy → IR adapters
# -----------------------------------------------------------------------


def keras_model_to_ir(
    parsed_model: Any,
    data_format: Optional[str] = None,
) -> IRModel:
    """Convert a parsed Keras model to the framework-agnostic IR.

    This adapter enables incremental migration: existing parsers still
    produce ``keras.models.Model`` objects, but callers can convert them
    to :class:`IRModel` before passing to new-style backends.

    Parameters
    ----------
    parsed_model
        A ``keras.models.Model`` instance.
    data_format
        ``'channels_first'`` or ``'channels_last'``.

    Returns
    -------
    IRModel
    """

    ir_layers: list[IRLayer] = []

    for layer in parsed_model.layers:
        layer_type_str = type(layer).__name__
        layer_type = LAYER_TYPE_FROM_STRING.get(layer_type_str)
        if layer_type is None:
            continue

        # --- weights ---
        weights = None
        w = layer.get_weights()
        if len(w) >= 2:
            weights = LayerWeights(
                kernel=w[0],
                bias=w[1],
                mask=w[2] if len(w) > 2 else None,
            )

        # --- layer-type-specific attributes ---
        kwargs: dict[str, Any] = {}

        if hasattr(layer, 'kernel_size') and layer.kernel_size is not None:
            kwargs['kernel_size'] = tuple(layer.kernel_size)
        if hasattr(layer, 'strides') and layer.strides is not None:
            kwargs['strides'] = tuple(layer.strides)
        if hasattr(layer, 'padding') and isinstance(layer.padding, str):
            kwargs['padding'] = layer.padding
        if hasattr(layer, 'filters') and layer.filters is not None:
            kwargs['filters'] = layer.filters
        if hasattr(layer, 'depth_multiplier'):
            kwargs['depth_multiplier'] = layer.depth_multiplier
        if hasattr(layer, 'dilation_rate') and layer.dilation_rate is not None:
            kwargs['dilation_rate'] = tuple(layer.dilation_rate)
        if hasattr(layer, 'pool_size') and layer.pool_size is not None:
            kwargs['pool_size'] = tuple(layer.pool_size)
        fmt = getattr(layer, 'data_format', None)
        if fmt in ('channels_first', 'channels_last'):
            kwargs['data_format'] = DataFormat(fmt)
        if hasattr(layer, 'axis') and layer_type == LayerType.CONCATENATE:
            kwargs['axis'] = layer.axis

        # --- activation ---
        if hasattr(layer, 'activation') and layer.activation is not None:
            act = layer.activation
            kwargs['activation'] = (
                act.__name__ if callable(act) else str(act)
            )

        # --- inbound layers ---
        inbound: tuple[str, ...] = ()
        try:
            nodes = layer._inbound_nodes
            if nodes:
                node = nodes[0]
                # Keras 2 had ``inbound_layers``; Keras 3 replaced it with
                # ``parent_nodes`` whose ``operation`` is the source layer.
                inbound_layers = getattr(node, 'inbound_layers', None)
                if inbound_layers is None:
                    parent_nodes = getattr(node, 'parent_nodes', ())
                    inbound_layers = [pn.operation for pn in parent_nodes]
                if not isinstance(inbound_layers, (list, tuple)):
                    inbound_layers = [inbound_layers]
                inbound = tuple(inb.name for inb in inbound_layers)
        except (AttributeError, IndexError):
            pass

        # --- shapes ---
        output_shape = _keras_output_shape(layer)
        input_shape = _keras_input_shape(layer)

        ir_layer = IRLayer(
            name=layer.name,
            layer_type=layer_type,
            output_shape=output_shape,
            inbound=inbound,
            weights=weights,
            extra_config={'input_shape': input_shape},
            **kwargs,
        )
        ir_layers.append(ir_layer)

    # Determine model input shape (without batch dim).  Keras 3 replaced
    # ``InputLayer.input_shape`` with ``batch_shape``.
    input_layer = parsed_model.layers[0]
    raw = getattr(input_layer, 'batch_shape', None)
    if raw is None:
        raw = getattr(input_layer, 'input_shape', None)
    if raw is None and hasattr(input_layer, 'output'):
        raw = tuple(input_layer.output.shape)
    if isinstance(raw, list):
        raw = raw[0]
    model_input_shape = tuple(int(x) for x in raw[1:])

    # Resolve the data format: prefer an explicit argument, otherwise pick
    # the first spatial layer's data_format, otherwise fall back to Keras's
    # global default.
    if data_format is None:
        data_format = next(
            (
                layer.data_format.value
                for layer in ir_layers
                if layer.data_format is not None
            ),
            'channels_last',
        )

    return IRModel(
        layers=ir_layers,
        input_shape=model_input_shape,
        data_format=DataFormat(data_format),
        original_framework='keras',
    )


def layer_list_to_ir(
    layer_list: list[dict],
    input_shape: tuple[int, ...],
    data_format: str = 'channels_last',
) -> IRModel:
    """Convert the legacy ``_layer_list`` directly to an :class:`IRModel`.

    This is the more efficient path: parsers can produce IR directly from
    ``_layer_list`` without the intermediate ``keras.models.Model`` step.

    Parameters
    ----------
    layer_list
        ``AbstractModelParser._layer_list`` — a list of dicts with keys
        like ``layer_type``, ``name``, ``inbound``, ``parameters``, etc.
    input_shape
        Network input shape *without* the batch dimension.
    data_format
        ``'channels_first'`` or ``'channels_last'``.

    Returns
    -------
    IRModel
    """

    ir_layers: list[IRLayer] = []

    # Synthetic input layer
    ir_layers.append(IRLayer(
        name='input',
        layer_type=LayerType.INPUT,
        output_shape=(None,) + tuple(input_shape),
    ))

    for entry in layer_list:
        d = dict(entry)  # shallow copy to avoid mutating caller's data

        layer_type_str = d.pop('layer_type', '')
        layer_type = LAYER_TYPE_FROM_STRING.get(layer_type_str)
        if layer_type is None:
            continue

        name = d.pop('name', '')
        inbound = tuple(d.pop('inbound', ()))

        # --- weights ---
        weights = None
        params = d.pop('parameters', d.pop('weights', None))
        if params is not None and len(params) >= 2:
            weights = LayerWeights(
                kernel=params[0],
                bias=params[1],
                mask=params[2] if len(params) > 2 else None,
            )

        activation = d.pop('activation', 'linear')
        if callable(activation):
            activation = getattr(activation, '__name__', str(activation))

        # Extract known fields; everything else goes to extra_config
        kernel_size = _to_tuple(d.pop('kernel_size', None))
        strides = _to_tuple(d.pop('strides', None))
        padding = d.pop('padding', 'valid')
        filters = d.pop('filters', None)
        depth_multiplier = d.pop('depth_multiplier', 1)
        dilation_rate = _to_tuple(d.pop('dilation_rate', None))
        pool_size = _to_tuple(d.pop('pool_size', None))
        axis = d.pop('axis', -1)
        target_shape = _to_tuple(d.pop('target_shape', None))
        output_shape = _to_tuple(d.pop('output_shape', (None,)))

        layer_data_format = d.pop('data_format', data_format)
        if isinstance(layer_data_format, str):
            layer_data_format = DataFormat(layer_data_format)

        ir_layer = IRLayer(
            name=name,
            layer_type=layer_type,
            output_shape=output_shape if output_shape else (None,),
            inbound=inbound,
            weights=weights,
            activation=activation,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            filters=filters,
            depth_multiplier=depth_multiplier,
            dilation_rate=dilation_rate,
            pool_size=pool_size,
            axis=axis,
            data_format=layer_data_format,
            target_shape=target_shape,
            extra_config=d,
        )
        ir_layers.append(ir_layer)

    return IRModel(
        layers=ir_layers,
        input_shape=tuple(input_shape),
        data_format=DataFormat(data_format),
    )


# -----------------------------------------------------------------------
# IR → Legacy adapters (facades)
# -----------------------------------------------------------------------


class _ActivationProxy:
    """Mimics a Keras activation function object with a ``__name__``."""

    def __init__(self, name: str) -> None:
        self.__name__ = name

    def __call__(self, x: Any) -> Any:  # pragma: no cover
        raise NotImplementedError(
            'IRLayerFacade activation proxies are not callable.'
        )


class IRLayerFacade:
    """Wraps an :class:`IRLayer` to expose a Keras-layer-like interface.

    Legacy ``AbstractSNN`` subclasses call ``layer.get_weights()``,
    ``layer.output_shape``, ``layer.kernel_size``, etc.  This facade
    provides those attributes so that old backends work transparently
    with :class:`IRModel`.

    ``type(facade).__name__`` returns the legacy string name (e.g.
    ``'Conv2D'``) so that ``get_type(layer)`` still works.
    """

    def __init__(self, ir_layer: IRLayer) -> None:
        self._ir = ir_layer

    # Make get_type(layer) work — it does layer.__class__.__name__
    @property
    def __class__(self) -> type:
        # Dynamically create a type with the right __name__
        type_name = self._ir.type_string
        return type(type_name, (), {})

    @property
    def name(self) -> str:
        return self._ir.name

    @property
    def output_shape(self) -> tuple:
        return self._ir.output_shape

    @property
    def input_shape(self) -> Optional[tuple]:
        return self._ir.input_shape

    @property
    def kernel_size(self) -> Optional[tuple[int, ...]]:
        return self._ir.kernel_size

    @property
    def strides(self) -> Optional[tuple[int, ...]]:
        return self._ir.strides

    @property
    def padding(self) -> str:
        return self._ir.padding

    @property
    def filters(self) -> Optional[int]:
        return self._ir.filters

    @property
    def depth_multiplier(self) -> int:
        return self._ir.depth_multiplier

    @property
    def dilation_rate(self) -> Optional[tuple[int, ...]]:
        return self._ir.dilation_rate

    @property
    def pool_size(self) -> Optional[tuple[int, ...]]:
        return self._ir.pool_size

    @property
    def data_format(self) -> str:
        return self._ir.data_format.value

    @property
    def activation(self) -> _ActivationProxy:
        return _ActivationProxy(self._ir.activation)

    @property
    def axis(self) -> int:
        return self._ir.axis

    @property
    def weights(self) -> list:
        """Mimics ``keras.layers.Layer.weights`` (list of variables)."""
        return self.get_weights()

    @property
    def bias(self) -> Optional[np.ndarray]:
        if self._ir.weights is not None:
            return self._ir.weights.bias
        return None

    def get_weights(self) -> list[np.ndarray]:
        """Mimics ``keras.layers.Layer.get_weights()``."""
        if self._ir.weights is None:
            return []
        return list(self._ir.weights.as_tuple())

    def get_config(self) -> dict[str, Any]:
        """Mimics ``keras.layers.Layer.get_config()``."""
        config: dict[str, Any] = {'name': self.name}
        if self._ir.kernel_size is not None:
            config['kernel_size'] = self._ir.kernel_size
        if self._ir.strides is not None:
            config['strides'] = self._ir.strides
        if self._ir.filters is not None:
            config['filters'] = self._ir.filters
        if self._ir.pool_size is not None:
            config['pool_size'] = self._ir.pool_size
        config['padding'] = self._ir.padding
        config['activation'] = self._ir.activation
        config.update(self._ir.extra_config)
        return config


class IRModelFacade:
    """Wraps an :class:`IRModel` to look like a ``keras.models.Model``.

    Provides the ``.layers``, ``.input_shape``, and ``.output_shape``
    attributes that ``AbstractSNN.build()``, ``setup_layers()``,
    ``init_log_vars()``, and ``set_connectivity()`` depend on.
    """

    def __init__(self, ir_model: IRModel) -> None:
        self._ir = ir_model
        self.layers: list[IRLayerFacade] = [
            IRLayerFacade(layer) for layer in ir_model.layers
        ]

    @property
    def input_shape(self) -> tuple:
        return (None,) + self._ir.input_shape

    @property
    def output_shape(self) -> tuple:
        return self._ir.output_layer.output_shape


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _to_tuple(value: Any) -> Optional[tuple]:
    """Coerce *value* to a tuple, or return ``None``."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _keras_output_shape(layer: Any) -> tuple:
    """Keras-3-compatible fetch of a layer's output shape."""
    out = getattr(layer, 'output', None)
    if out is not None and hasattr(out, 'shape'):
        return _safe_shape(tuple(out.shape))
    return _safe_shape(getattr(layer, 'output_shape', None))


def _keras_input_shape(layer: Any) -> tuple:
    """Keras-3-compatible fetch of a layer's input shape (or ``()``)."""
    inp = getattr(layer, 'input', None)
    if inp is not None and hasattr(inp, 'shape'):
        return _safe_shape(tuple(inp.shape))
    legacy = getattr(layer, 'input_shape', None)
    if legacy is not None:
        return _safe_shape(legacy)
    batch_shape = getattr(layer, 'batch_shape', None)
    if batch_shape is not None:
        return _safe_shape(batch_shape)
    return ()


def _safe_shape(shape: Any) -> tuple:
    """Normalize a shape to a plain tuple of ints/None."""
    if shape is None:
        return ()
    if isinstance(shape, list):
        shape = shape[0] if len(shape) == 1 else tuple(shape)
    return tuple(
        int(x) if x is not None and x != 'None' else None
        for x in shape
    )
