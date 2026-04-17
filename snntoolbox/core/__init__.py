"""Core module providing framework-agnostic interfaces for snn_toolbox.

Public API
----------
.. autosummary::

    ir.IRModel
    ir.IRLayer
    ir.LayerWeights
    ir.LayerType
    ir.DataFormat
    protocols.ModelParserBase
    protocols.SNNBackendBase
    protocols.TelemetryHook
    protocols.NullTelemetry
    registry.parser_registry
    registry.backend_registry
    adapters.keras_model_to_ir
    adapters.layer_list_to_ir
    adapters.IRModelFacade
    adapters.IRLayerFacade
    connectivity.compute_connectivity
    connectivity.ConnectivityStats
"""

from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
)
from snntoolbox.core.protocols import (
    ModelParserBase,
    NullTelemetry,
    SNNBackendBase,
    TelemetryHook,
)
from snntoolbox.core.registry import backend_registry, parser_registry

__all__ = [
    'DataFormat',
    'IRLayer',
    'IRModel',
    'LayerType',
    'LayerWeights',
    'ModelParserBase',
    'NullTelemetry',
    'SNNBackendBase',
    'TelemetryHook',
    'backend_registry',
    'parser_registry',
]
