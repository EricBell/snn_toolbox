"""Framework-agnostic Intermediate Representation for neural network models.

This module defines the data structures that decouple model parsers from
spiking backends. Instead of passing a ``keras.models.Model`` between
pipeline stages, parsers produce an :class:`IRModel` and backends consume it.

The IR captures everything that spiking backends currently extract from
Keras layer objects: weights, shapes, connectivity, and layer-type-specific
attributes (kernel_size, strides, padding, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class LayerType(Enum):
    """Supported neural network layer types.

    Maps 1:1 to the string layer types used in ``_layer_list['layer_type']``
    and ``get_type()`` throughout the legacy codebase.
    """

    INPUT = auto()
    DENSE = auto()
    CONV1D = auto()
    CONV2D = auto()
    CONV2D_TRANSPOSE = auto()
    DEPTHWISE_CONV2D = auto()
    AVERAGE_POOLING_2D = auto()
    MAX_POOLING_2D = auto()
    FLATTEN = auto()
    RESHAPE = auto()
    CONCATENATE = auto()
    ZERO_PADDING_2D = auto()
    UPSAMPLING_2D = auto()
    # Sparse variants (used with keras_rewiring)
    SPARSE = auto()
    SPARSE_CONV2D = auto()
    SPARSE_DEPTHWISE_CONV2D = auto()


#: Maps legacy string type names to LayerType enum members.
LAYER_TYPE_FROM_STRING: dict[str, LayerType] = {
    'InputLayer': LayerType.INPUT,
    'Dense': LayerType.DENSE,
    'Conv1D': LayerType.CONV1D,
    'Conv2D': LayerType.CONV2D,
    'Conv2DTranspose': LayerType.CONV2D_TRANSPOSE,
    'DepthwiseConv2D': LayerType.DEPTHWISE_CONV2D,
    'AveragePooling2D': LayerType.AVERAGE_POOLING_2D,
    'MaxPooling2D': LayerType.MAX_POOLING_2D,
    'Flatten': LayerType.FLATTEN,
    'Reshape': LayerType.RESHAPE,
    'Concatenate': LayerType.CONCATENATE,
    'ZeroPadding2D': LayerType.ZERO_PADDING_2D,
    'UpSampling2D': LayerType.UPSAMPLING_2D,
    'Sparse': LayerType.SPARSE,
    'SparseConv2D': LayerType.SPARSE_CONV2D,
    'SparseDepthwiseConv2D': LayerType.SPARSE_DEPTHWISE_CONV2D,
}

#: Reverse mapping from LayerType enum to legacy string names.
LAYER_TYPE_TO_STRING: dict[LayerType, str] = {
    v: k for k, v in LAYER_TYPE_FROM_STRING.items()
}


class DataFormat(Enum):
    """Channel ordering for spatial data."""

    CHANNELS_FIRST = 'channels_first'
    CHANNELS_LAST = 'channels_last'


@dataclass(frozen=True)
class LayerWeights:
    """Framework-agnostic weight container for a single layer.

    Parameters
    ----------
    kernel : np.ndarray
        Weight matrix / convolution kernel.
    bias : np.ndarray
        Bias vector.
    mask : np.ndarray, optional
        Sparsity mask for sparse layers (from keras_rewiring).
    """

    kernel: np.ndarray
    bias: np.ndarray
    mask: Optional[np.ndarray] = None

    def as_tuple(self) -> tuple:
        """Return weights in the (kernel, bias[, mask]) tuple format
        expected by legacy code that destructures ``layer.get_weights()``."""
        if self.mask is not None:
            return (self.kernel, self.bias, self.mask)
        return (self.kernel, self.bias)


@dataclass(frozen=True)
class IRLayer:
    """Framework-agnostic description of a single neural network layer.

    Captures every attribute that spiking backends currently extract from
    ``keras.layers.Layer`` objects: name, type, shape, weights, activation,
    and layer-type-specific parameters (kernel_size, strides, etc.).

    Parameters
    ----------
    name : str
        Unique layer name (e.g. ``'00Dense_128'``).
    layer_type : LayerType
        The kind of layer.
    output_shape : tuple[int, ...]
        Layer output shape *including* the batch dimension (first element may
        be ``None``).
    inbound : tuple[str, ...]
        Names of layers whose outputs feed into this layer.
    weights : LayerWeights, optional
        Trainable parameters (``None`` for layers without weights).
    activation : str
        Activation function name (``'relu'``, ``'softmax'``, ``'linear'``, ...).
    """

    name: str
    layer_type: LayerType
    output_shape: tuple

    # Connectivity
    inbound: tuple[str, ...] = ()

    # Weights
    weights: Optional[LayerWeights] = None

    # Activation
    activation: str = 'linear'

    # Convolution / transpose-convolution attributes
    kernel_size: Optional[tuple[int, ...]] = None
    strides: Optional[tuple[int, ...]] = None
    padding: str = 'valid'
    filters: Optional[int] = None
    depth_multiplier: int = 1
    dilation_rate: Optional[tuple[int, ...]] = None

    # Pooling attributes
    pool_size: Optional[tuple[int, ...]] = None

    # Concatenate axis
    axis: int = -1

    # Spatial data format
    data_format: DataFormat = DataFormat.CHANNELS_LAST

    # Reshape target
    target_shape: Optional[tuple[int, ...]] = None

    # ZeroPadding size
    padding_size: Optional[tuple] = None

    # Escape hatch for framework-specific or future attributes
    extra_config: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def input_shape(self) -> Optional[tuple]:
        """Input shape stored during conversion (via ``extra_config``)."""
        return self.extra_config.get('input_shape')

    @property
    def has_weights(self) -> bool:
        return self.weights is not None

    @property
    def num_neurons(self) -> int:
        """Total neuron count (product of spatial dims, excluding batch)."""
        return int(np.prod(self.output_shape[1:]))

    @property
    def type_string(self) -> str:
        """Legacy string name (e.g. ``'Conv2D'``, ``'Dense'``)."""
        return LAYER_TYPE_TO_STRING.get(self.layer_type, self.layer_type.name)


@dataclass
class IRModel:
    """Framework-agnostic intermediate representation of a neural network.

    Replaces the pattern where the parsed model IS a ``keras.models.Model``.
    This is a pure data structure consumable without importing any ML framework.

    Parameters
    ----------
    layers : list[IRLayer]
        Ordered list of layers (first is input, last is output).
    input_shape : tuple[int, ...]
        Network input shape *without* the batch dimension.
    data_format : DataFormat
        Channel ordering for the model's spatial layers.
    name : str
        Optional model name.
    original_framework : str
        Framework the model was originally defined in (e.g. ``'keras'``).
    """

    layers: list[IRLayer]
    input_shape: tuple[int, ...]
    data_format: DataFormat = DataFormat.CHANNELS_LAST
    name: str = ''
    original_framework: str = ''

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_layer(self, name: str) -> Optional[IRLayer]:
        """Return the layer with the given *name*, or ``None``."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    @property
    def input_layer(self) -> IRLayer:
        return self.layers[0]

    @property
    def output_layer(self) -> IRLayer:
        return self.layers[-1]

    @property
    def num_classes(self) -> int:
        """Number of output classes (last dim of the output layer)."""
        return self.output_layer.output_shape[-1]

    def layers_with_weights(self) -> list[IRLayer]:
        """Return only layers that carry trainable weights."""
        return [layer for layer in self.layers if layer.has_weights]

    def get_inbound_layers(self, layer: IRLayer) -> list[IRLayer]:
        """Resolve *layer*'s inbound names to ``IRLayer`` objects."""
        result = []
        for name in layer.inbound:
            inbound = self.get_layer(name)
            if inbound is not None:
                result.append(inbound)
        return result
