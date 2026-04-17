"""Formal interfaces for the snn_toolbox plugin system.

This module provides:

* **Protocols** (structural subtyping) for lightweight contracts:
  :class:`ToolboxConfig`, :class:`WeightNormalizer`, :class:`TelemetryHook`.

* **Abstract Base Classes** for the two core pipeline components:
  :class:`ModelParserBase` (ANN ingestion) and :class:`SNNBackendBase`
  (spiking simulation).  These enforce strict interfaces via
  ``@abstractmethod``, include modern :mod:`logging` in every concrete
  method, and accept a pluggable :class:`TelemetryHook` for timing and
  metrics.

Telemetry
---------
Every ``on_phase_start`` / ``on_phase_end`` pair brackets a logical unit of
work (parsing, building, simulating a batch, …).  ``on_metric`` reports
scalar measurements (accuracy, spike rate, operation count) that downstream
observers (W&B, MLflow, a simple CSV logger, …) can record.

:class:`NullTelemetry` is the zero-cost default — all methods are no-ops.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from snntoolbox.core.ir import IRLayer, IRModel, LayerType

# ---------------------------------------------------------------------------
# Protocols (structural subtyping)
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolboxConfig(Protocol):
    """Typed interface over ``configparser.ConfigParser``.

    Every pipeline component receives a config object.  This protocol
    documents the subset of the ConfigParser API that is actually used,
    enabling alternative implementations (e.g. a dataclass-backed config).
    """

    def get(self, section: str, key: str) -> str: ...
    def getint(self, section: str, key: str) -> int: ...
    def getfloat(self, section: str, key: str) -> float: ...
    def getboolean(self, section: str, key: str) -> bool: ...
    def set(self, section: str, key: str, value: str) -> None: ...


@runtime_checkable
class WeightNormalizer(Protocol):
    """Protocol for weight normalization strategies.

    Normalization adjusts layer weights so that the SNN's firing rates
    approximate the ANN's activations.  Implementations receive an
    :class:`~snntoolbox.core.ir.IRModel` and return a *new* ``IRModel``
    with updated weights (the original is not mutated).
    """

    def normalize(
        self,
        ir_model: IRModel,
        config: ToolboxConfig,
        x_norm: Optional[np.ndarray] = None,
        dataflow: Any = None,
    ) -> IRModel:
        """Normalize weights, returning a new :class:`IRModel`."""
        ...


@runtime_checkable
class TelemetryHook(Protocol):
    """Pluggable observer for timing, metrics, and lifecycle events.

    Implementations might log to W&B, MLflow, a CSV file, or stdout.
    The default :class:`NullTelemetry` makes all calls no-ops.
    """

    def on_phase_start(
        self,
        phase: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called when a named phase begins (e.g. ``'parse'``, ``'build'``)."""
        ...

    def on_phase_end(
        self,
        phase: str,
        metadata: Optional[dict[str, Any]] = None,
        elapsed_sec: float = 0.0,
    ) -> None:
        """Called when a named phase ends."""
        ...

    def on_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Report a scalar metric (accuracy, spike rate, op count, …)."""
        ...


class NullTelemetry:
    """Default telemetry hook — every method is a no-op."""

    def on_phase_start(
        self,
        phase: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        pass

    def on_phase_end(
        self,
        phase: str,
        metadata: Optional[dict[str, Any]] = None,
        elapsed_sec: float = 0.0,
    ) -> None:
        pass

    def on_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        pass


_NULL_TELEMETRY = NullTelemetry()

# ---------------------------------------------------------------------------
# ModelParserBase — Abstract Base Class
# ---------------------------------------------------------------------------


class ModelParserBase(ABC):
    """Abstract base class for neural network model parsers.

    Subclasses implement framework-specific extraction methods.  The
    concrete :meth:`parse` template method orchestrates them and returns
    an :class:`~snntoolbox.core.ir.IRModel`.

    Parameters
    ----------
    input_model
        The framework-specific model object (e.g. a Keras ``Model``).
    config
        Toolbox configuration for this experiment.
    logger
        A :class:`logging.Logger`.  If *None*, a child logger under
        ``snntoolbox.parsing`` is created.
    telemetry
        A :class:`TelemetryHook` implementation.  Defaults to
        :class:`NullTelemetry`.
    """

    def __init__(
        self,
        input_model: Any,
        config: ToolboxConfig,
        *,
        logger: Optional[logging.Logger] = None,
        telemetry: Optional[TelemetryHook] = None,
    ) -> None:
        self.input_model = input_model
        self.config = config
        self.logger = logger or logging.getLogger('snntoolbox.parsing')
        self.telemetry: TelemetryHook = telemetry or _NULL_TELEMETRY
        self._layer_list: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def get_layer_iterable(self) -> Sequence:
        """Return an iterable over the layers of the input model."""

    @abstractmethod
    def get_type(self, layer: Any) -> str:
        """Return the layer's class name (e.g. ``'Conv2D'``)."""

    @abstractmethod
    def get_batchnorm_parameters(self, layer: Any) -> tuple:
        """Extract batch-normalization parameters.

        Returns
        -------
        mean, var_eps_sqrt_inv, gamma, beta, axis : tuple
        """

    @abstractmethod
    def get_inbound_layers(self, layer: Any) -> list:
        """Return the inbound layers of *layer*."""

    @abstractmethod
    def has_weights(self, layer: Any) -> bool:
        """Return ``True`` if *layer* has trainable parameters."""

    @abstractmethod
    def get_input_shape(self) -> tuple[int, ...]:
        """Network input shape, *excluding* the batch dimension."""

    @abstractmethod
    def get_output_shape(self, layer: Any) -> tuple[int, ...]:
        """Output shape of *layer*, *including* the batch dimension."""

    @abstractmethod
    def parse_dense(self, layer: Any, attributes: dict) -> None:
        """Populate *attributes* with Dense-layer specifics."""

    @abstractmethod
    def parse_convolution(self, layer: Any, attributes: dict) -> None:
        """Populate *attributes* with Conv-layer specifics."""

    @abstractmethod
    def parse_depthwiseconvolution(self, layer: Any, attributes: dict) -> None:
        """Populate *attributes* with DepthwiseConv2D specifics."""

    @abstractmethod
    def parse_pooling(self, layer: Any, attributes: dict) -> None:
        """Populate *attributes* with pooling-layer specifics."""

    @abstractmethod
    def get_activation(self, layer: Any) -> str:
        """Return the activation function name for *layer*."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def initialize_attributes(self, layer: Any = None) -> dict[str, Any]:
        """Return a fresh attributes dict for a layer being parsed."""
        return {}

    def parse_sparse(self, layer: Any, attributes: dict) -> None:
        """Populate *attributes* for a Sparse layer. Override if supported."""
        raise NotImplementedError(
            f'{type(self).__name__} does not support sparse layers.'
        )

    def parse_concatenate(self, layer: Any, attributes: dict) -> None:
        """Populate *attributes* for a Concatenate layer."""

    @property
    def layers_to_skip(self) -> list[str]:
        """Layer types removed during parsing (training-only layers, etc.)."""
        return [
            'BatchNormalization',
            'Activation',
            'Dropout',
            'ReLU',
            'ActivityRegularization',
            'GaussianNoise',
        ]

    # ------------------------------------------------------------------
    # Concrete template methods (logging + telemetry)
    # ------------------------------------------------------------------

    def parse(self) -> IRModel:
        """Parse the input model into a framework-agnostic IR.

        Iterates over layers, absorbs batch-norm, skips training-only
        layers, and collects layer specifications into ``_layer_list``.
        Returns an :class:`~snntoolbox.core.ir.IRModel`.

        This is the main entry point — subclasses should *not* override it.
        Override the individual ``parse_*`` / ``get_*`` methods instead.
        """
        from snntoolbox.core.adapters import layer_list_to_ir

        self.telemetry.on_phase_start('parse', {
            'model_type': type(self.input_model).__name__,
        })
        t0 = time.perf_counter()

        self._layer_list.clear()
        snn_layers = eval(self.config.get('restrictions', 'snn_layers'))

        layers = self.get_layer_iterable()
        name_map: dict[str, int] = {}
        idx = 0
        inserted_flatten = False
        skipped = 0

        for layer in layers:
            layer_type = self.get_type(layer)

            if layer_type in self.layers_to_skip:
                self.logger.debug('Skipping %s layer.', layer_type)
                skipped += 1
                continue

            if layer_type not in snn_layers:
                self.logger.debug(
                    'Layer type %s not in snn_layers, skipping.', layer_type,
                )
                skipped += 1
                continue

            self.logger.info('Parsing layer %s (idx=%d).', layer_type, idx)

            attributes = self.initialize_attributes(layer)
            attributes.update({
                'layer_type': layer_type,
                'name': self._make_name(layer, idx),
                'inbound': self._get_inbound_names(layer, name_map),
            })

            self._dispatch_parse(layer_type, layer, attributes)

            self._layer_list.append(attributes)
            name_map[str(id(layer))] = idx
            idx += 1

        input_shape = self.get_input_shape()
        data_format = self.config.get('input', 'data_format') \
            if self.config.get('input', 'data_format') \
            else 'channels_last'

        ir_model = layer_list_to_ir(
            self._layer_list, input_shape, data_format,
        )

        elapsed = time.perf_counter() - t0
        self.logger.info(
            'Parsing complete: %d layers parsed, %d skipped (%.3fs).',
            idx, skipped, elapsed,
        )
        self.telemetry.on_phase_end('parse', {
            'num_layers_parsed': idx,
            'num_layers_skipped': skipped,
        }, elapsed_sec=elapsed)
        self.telemetry.on_metric('parse/num_layers', float(idx))
        self.telemetry.on_metric('parse/elapsed_sec', elapsed)

        return ir_model

    def evaluate(
        self,
        val_fn: Any,
        batch_size: int,
        num_to_test: int,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        dataflow: Any = None,
    ) -> float:
        """Evaluate model accuracy.  Override for framework-specific logic."""
        raise NotImplementedError(
            f'{type(self).__name__} does not implement evaluate().'
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_parse(
        self,
        layer_type: str,
        layer: Any,
        attributes: dict,
    ) -> None:
        """Route to the correct ``parse_*`` method."""
        if layer_type == 'Dense':
            self.parse_dense(layer, attributes)
        elif layer_type in {'Conv1D', 'Conv2D'}:
            self.parse_convolution(layer, attributes)
        elif layer_type == 'DepthwiseConv2D':
            self.parse_depthwiseconvolution(layer, attributes)
        elif layer_type in {'AveragePooling2D', 'MaxPooling2D'}:
            self.parse_pooling(layer, attributes)
        elif layer_type == 'Sparse':
            self.parse_sparse(layer, attributes)
        elif layer_type == 'Concatenate':
            self.parse_concatenate(layer, attributes)

    def _make_name(self, layer: Any, idx: int) -> str:
        """Build a name like ``'00Conv2D_32x64x64'``."""
        layer_type = self.get_type(layer)
        output_shape = self.get_output_shape(layer)
        shape_parts = [f'{x}x' for x in output_shape[1:]]
        if shape_parts:
            shape_parts[0] = '_' + shape_parts[0]
            shape_parts[-1] = shape_parts[-1].rstrip('x')
        shape_string = ''.join(shape_parts)
        num_str = str(idx).zfill(2)
        return f'{num_str}{layer_type}{shape_string}'

    def _get_inbound_names(
        self, layer: Any, name_map: dict[str, int],
    ) -> list[str]:
        """Resolve inbound layer references to our internal names."""
        inbound = self.get_inbound_layers(layer)
        for i in range(len(inbound)):
            for _ in range(len(self.layers_to_skip)):
                if self.get_type(inbound[i]) in self.layers_to_skip:
                    inbound[i] = self.get_inbound_layers(inbound[i])[0]
                else:
                    break
        if not self._layer_list or any(
            self.get_type(inb) == 'InputLayer' for inb in inbound
        ):
            return ['input']
        return [
            self._layer_list[name_map[str(id(inb))]]['name']
            for inb in inbound
        ]


# ---------------------------------------------------------------------------
# SNNBackendBase — Abstract Base Class
# ---------------------------------------------------------------------------


class SNNBackendBase(ABC):
    """Abstract base class for spiking neural network simulators.

    Subclasses implement simulator-specific neuron / synapse creation.
    The concrete :meth:`build` and :meth:`run` template methods
    orchestrate them with logging and telemetry.

    Parameters
    ----------
    config
        Toolbox configuration for this experiment.
    queue
        Optional queue for GUI stop-signal detection.
    logger
        A :class:`logging.Logger`.  If *None*, a child logger under
        ``snntoolbox.simulation`` is created.
    telemetry
        A :class:`TelemetryHook` implementation.  Defaults to
        :class:`NullTelemetry`.
    """

    def __init__(
        self,
        config: ToolboxConfig,
        *,
        queue: Any = None,
        logger: Optional[logging.Logger] = None,
        telemetry: Optional[TelemetryHook] = None,
    ) -> None:
        self.config = config
        self.queue = queue
        self.logger = logger or logging.getLogger('snntoolbox.simulation')
        self.telemetry: TelemetryHook = telemetry or _NULL_TELEMETRY

        self.ir_model: Optional[IRModel] = None
        self.is_built: bool = False

        # Simulation parameters
        self._dt: float = config.getfloat('simulation', 'dt')
        self._duration: int = config.getint('simulation', 'duration')
        self._num_timesteps: int = int(self._duration / self._dt)

        # Batch size
        self._batch_size: int = config.getint('simulation', 'batch_size')
        self.batch_size: int = self._adjust_batchsize()

        # Network topology metrics (populated during build)
        self.num_classes: Optional[int] = None
        self.num_neurons: Optional[list[int]] = None
        self.num_synapses: Optional[int] = None

        # Telemetry accumulators
        self._build_elapsed: float = 0.0

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by subclasses
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_parallelizable(self) -> bool:
        """Whether the simulator can test multiple samples in parallel."""

    @abstractmethod
    def add_input_layer(self, input_shape: tuple[int, ...]) -> None:
        """Add the input layer with the given *input_shape* (incl. batch)."""

    @abstractmethod
    def add_layer(self, layer: IRLayer) -> None:
        """Perform any per-layer setup independent of layer type."""

    @abstractmethod
    def build_dense(self, layer: IRLayer) -> None:
        """Build a spiking fully-connected layer from *layer*."""

    @abstractmethod
    def build_convolution(self, layer: IRLayer) -> None:
        """Build a spiking convolutional layer from *layer*."""

    @abstractmethod
    def build_pooling(self, layer: IRLayer) -> None:
        """Build a spiking pooling layer from *layer*."""

    @abstractmethod
    def compile(self) -> None:
        """Compile the spiking network after all layers are added."""

    @abstractmethod
    def simulate(self, **kwargs: Any) -> np.ndarray:
        """Simulate one batch for the configured duration.

        Returns
        -------
        output_b_l_t : np.ndarray
            Shape ``(batch_size, num_classes, num_timesteps)``.
        """

    @abstractmethod
    def reset(self, sample_idx: int) -> None:
        """Reset network state between samples or batches."""

    @abstractmethod
    def end_sim(self) -> None:
        """Release simulator resources."""

    @abstractmethod
    def save(self, path: str, filename: str) -> None:
        """Serialize the spiking model to disk."""

    @abstractmethod
    def load(self, path: str, filename: str) -> None:
        """Restore a spiking model from disk."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def build_flatten(self, layer: IRLayer) -> None:
        """Build a flatten layer.  No-op by default."""

    def init_cells(self) -> None:
        """Initialize neuron cell parameters.  No-op by default."""

    def get_spiketrains(self, **kwargs: Any) -> Optional[np.ndarray]:
        """Return spike trains of a layer.  ``None`` by default."""
        return None

    def preprocessing(self, **kwargs: Any) -> None:
        """Preprocessing hook called before ``setup_layers``.  No-op."""

    # ------------------------------------------------------------------
    # Concrete template methods (logging + telemetry)
    # ------------------------------------------------------------------

    def build(self, ir_model: IRModel, **kwargs: Any) -> None:
        """Assemble the spiking network from an :class:`IRModel`.

        Orchestrates ``add_input_layer`` → per-layer ``add_layer`` +
        ``build_*`` → ``compile``, all with logging and telemetry.
        """
        self.telemetry.on_phase_start('build', {
            'num_layers': len(ir_model.layers),
            'input_shape': ir_model.input_shape,
        })
        t0 = time.perf_counter()

        self.logger.info('Building spiking model...')
        self.ir_model = ir_model
        self.num_classes = ir_model.num_classes
        self.logger.debug(
            'Model: %d layers, %d output classes.',
            len(ir_model.layers), self.num_classes,
        )

        batch_shape = list((self.batch_size,) + ir_model.input_shape)

        self.preprocessing(**kwargs)
        self.setup_layers(ir_model, batch_shape)

        self.logger.info('Compiling spiking model...')
        self.compile()
        self.is_built = True

        elapsed = time.perf_counter() - t0
        self._build_elapsed = elapsed
        self.logger.info('Build complete (%.3fs).', elapsed)
        self.telemetry.on_phase_end('build', {
            'num_classes': self.num_classes,
        }, elapsed_sec=elapsed)
        self.telemetry.on_metric('build/elapsed_sec', elapsed)
        self.telemetry.on_metric(
            'build/num_layers', float(len(ir_model.layers)),
        )

    def setup_layers(
        self,
        ir_model: IRModel,
        batch_shape: list[int],
    ) -> None:
        """Iterate over IR layers and dispatch to ``build_*`` methods."""
        self.add_input_layer(tuple(batch_shape))

        for layer in ir_model.layers[1:]:
            self.logger.debug('Building layer: %s (%s)', layer.name,
                              layer.type_string)
            self.add_layer(layer)

            lt = layer.layer_type
            if lt == LayerType.DENSE:
                self.build_dense(layer)
            elif lt in (LayerType.CONV1D, LayerType.CONV2D,
                        LayerType.CONV2D_TRANSPOSE,
                        LayerType.DEPTHWISE_CONV2D):
                self.build_convolution(layer)
            elif lt in (LayerType.MAX_POOLING_2D,
                        LayerType.AVERAGE_POOLING_2D):
                self.build_pooling(layer)
            elif lt == LayerType.FLATTEN:
                self.build_flatten(layer)

    def run(
        self,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        dataflow: Any = None,
        **kwargs: Any,
    ) -> float:
        """Simulate the SNN on test data, returning top-1 accuracy.

        Handles batching, calls :meth:`simulate` per batch, computes
        accuracy, and reports metrics through logging and telemetry.
        """
        self.telemetry.on_phase_start('run')
        t0 = time.perf_counter()

        if x_test is None and dataflow is None:
            raise ValueError('Either x_test or dataflow must be provided.')

        num_to_test = self.config.getint('simulation', 'num_to_test')
        num_batches = num_to_test // self.batch_size

        truth_all: list[int] = []
        guesses_all: list[int] = []

        self.init_cells()

        for batch_idx in range(num_batches):
            # Fetch batch
            if x_test is not None:
                start = self.batch_size * batch_idx
                end = self.batch_size * (batch_idx + 1)
                x_b = x_test[start:end]
                y_b = y_test[start:end] if y_test is not None else None
            else:
                x_b, y_b = next(dataflow)

            if len(x_b) < self.batch_size:
                continue

            truth_b = np.argmax(y_b, axis=1)

            # Simulate
            self.telemetry.on_phase_start('simulate_batch', {
                'batch_idx': batch_idx,
            })
            t_batch = time.perf_counter()

            output_b_l_t = self.simulate(
                x_b_l=x_b, truth_b=truth_b, **kwargs,
            )

            batch_elapsed = time.perf_counter() - t_batch
            self.telemetry.on_phase_end(
                'simulate_batch',
                {'batch_idx': batch_idx},
                elapsed_sec=batch_elapsed,
            )

            # Classify
            guesses_b_t = np.argmax(output_b_l_t, 1)
            undecided = np.nonzero(np.sum(output_b_l_t, 1) == 0)
            guesses_b_t[undecided] = -1

            truth_all.extend(truth_b.tolist())
            guesses_all.extend(guesses_b_t[:, -1].tolist())

            # Running accuracy
            acc = np.mean(np.array(truth_all) == np.array(guesses_all))
            self.logger.info(
                'Batch %d/%d — running accuracy: %.2f%%',
                batch_idx + 1, num_batches, acc * 100,
            )
            self.telemetry.on_metric(
                'run/top1_accuracy', acc, step=batch_idx,
            )

            self.reset(batch_idx)

        total_acc = (
            np.mean(np.array(truth_all) == np.array(guesses_all))
            if truth_all else 0.0
        )
        elapsed = time.perf_counter() - t0

        self.logger.info(
            'Simulation complete: %.2f%% accuracy on %d samples (%.3fs).',
            total_acc * 100, len(guesses_all), elapsed,
        )
        self.telemetry.on_phase_end('run', {
            'total_accuracy': total_acc,
            'num_samples': len(guesses_all),
        }, elapsed_sec=elapsed)
        self.telemetry.on_metric('run/final_accuracy', total_acc)
        self.telemetry.on_metric('run/elapsed_sec', elapsed)

        return float(total_acc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjust_batchsize(self) -> int:
        """Reduce batch size to 1 if the simulator is not parallelizable."""
        bs = self._batch_size
        if bs > 1 and not self.is_parallelizable:
            self.logger.info(
                'Simulator is not parallelizable; setting batch_size=1.',
            )
            self.config.set('simulation', 'batch_size', '1')
            return 1
        return bs
