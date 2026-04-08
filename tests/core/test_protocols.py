"""Tests for the formal ABCs and telemetry hooks."""

import logging
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from snntoolbox.core.ir import IRLayer, IRModel, LayerType, LayerWeights
from snntoolbox.core.protocols import (
    ModelParserBase,
    NullTelemetry,
    SNNBackendBase,
    TelemetryHook,
)


# -----------------------------------------------------------------------
# Helpers: minimal concrete config stub
# -----------------------------------------------------------------------

class StubConfig:
    """Minimal config satisfying ToolboxConfig protocol."""

    def __init__(self, overrides: Optional[dict] = None):
        self._data = {
            ('simulation', 'dt'): '1.0',
            ('simulation', 'duration'): '100',
            ('simulation', 'batch_size'): '1',
            ('simulation', 'num_to_test'): '2',
            ('simulation', 'top_k'): '5',
            ('input', 'input_rate'): '1000',
            ('input', 'data_format'): 'channels_last',
            ('input', 'dataset_format'): 'npz',
            ('input', 'poisson_input'): 'false',
            ('input', 'num_poisson_events_per_sample'): '0',
            ('simulation', 'early_stopping'): 'false',
            ('restrictions', 'snn_layers'):
                "{'Dense', 'Conv2D', 'Flatten', 'AveragePooling2D'}",
            ('output', 'log_vars'): '{}',
            ('output', 'plot_vars'): '{}',
        }
        if overrides:
            self._data.update(overrides)

    def get(self, section: str, key: str) -> str:
        return self._data.get((section, key), '')

    def getint(self, section: str, key: str) -> int:
        return int(self._data[(section, key)])

    def getfloat(self, section: str, key: str) -> float:
        return float(self._data[(section, key)])

    def getboolean(self, section: str, key: str) -> bool:
        return self._data[(section, key)].lower() in ('true', '1', 'yes')

    def set(self, section: str, key: str, value: str) -> None:
        self._data[(section, key)] = value


# -----------------------------------------------------------------------
# Helpers: minimal concrete subclasses
# -----------------------------------------------------------------------

class ConcreteParser(ModelParserBase):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, layers, config, **kwargs):
        super().__init__(layers, config, **kwargs)
        self._layers = layers

    def get_layer_iterable(self):
        return self._layers

    def get_type(self, layer):
        return layer['type']

    def get_batchnorm_parameters(self, layer):
        return ()

    def get_inbound_layers(self, layer):
        return layer.get('inbound_refs', [])

    def has_weights(self, layer):
        return 'weights' in layer

    def get_input_shape(self):
        return (784,)

    def get_output_shape(self, layer):
        return layer.get('output_shape', (None, 10))

    def parse_dense(self, layer, attributes):
        attributes['parameters'] = layer.get('weights', (
            np.ones((784, 10), dtype='float32'),
            np.zeros(10, dtype='float32'),
        ))

    def parse_convolution(self, layer, attributes):
        pass

    def parse_depthwiseconvolution(self, layer, attributes):
        pass

    def parse_pooling(self, layer, attributes):
        pass

    def get_activation(self, layer):
        return layer.get('activation', 'relu')


class ConcreteBackend(SNNBackendBase):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, config, **kwargs):
        # Must set is_parallelizable before super().__init__ since it
        # calls _adjust_batchsize() which reads the property.
        self._parallelizable = True
        super().__init__(config, **kwargs)
        self._layers_built: list[str] = []

    @property
    def is_parallelizable(self) -> bool:
        return self._parallelizable

    def add_input_layer(self, input_shape):
        self._layers_built.append(f'input:{input_shape}')

    def add_layer(self, layer):
        self._layers_built.append(f'add:{layer.name}')

    def build_dense(self, layer):
        self._layers_built.append(f'dense:{layer.name}')

    def build_convolution(self, layer):
        self._layers_built.append(f'conv:{layer.name}')

    def build_pooling(self, layer):
        self._layers_built.append(f'pool:{layer.name}')

    def compile(self):
        self._layers_built.append('compile')

    def simulate(self, **kwargs):
        bs = self.batch_size
        nc = self.num_classes
        nt = self._num_timesteps
        out = np.zeros((bs, nc, nt))
        truth = kwargs.get('truth_b', np.zeros(bs, dtype=int))
        for i, t in enumerate(truth):
            out[i, int(t), :] = 1.0
        return out

    def reset(self, sample_idx):
        pass

    def end_sim(self):
        pass

    def save(self, path, filename):
        pass

    def load(self, path, filename):
        pass


# -----------------------------------------------------------------------
# Tests: TelemetryHook & NullTelemetry
# -----------------------------------------------------------------------

class TestTelemetry:

    def test_null_telemetry_is_noop(self):
        t = NullTelemetry()
        t.on_phase_start('test')
        t.on_phase_end('test', elapsed_sec=1.0)
        t.on_metric('acc', 0.95)

    def test_null_telemetry_satisfies_protocol(self):
        assert isinstance(NullTelemetry(), TelemetryHook)

    def test_custom_hook_satisfies_protocol(self):
        class MyHook:
            def on_phase_start(self, phase, metadata=None):
                pass
            def on_phase_end(self, phase, metadata=None, elapsed_sec=0.0):
                pass
            def on_metric(self, name, value, step=None):
                pass

        assert isinstance(MyHook(), TelemetryHook)


# -----------------------------------------------------------------------
# Tests: ModelParserBase
# -----------------------------------------------------------------------

class TestModelParserBase:

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError, match='abstract method'):
            ModelParserBase(None, StubConfig())

    def test_parse_returns_ir_model(self):
        layers = [
            {'type': 'Dense', 'output_shape': (None, 128),
             'weights': (np.ones((784, 128), 'f'), np.zeros(128, 'f')),
             'inbound_refs': []},
        ]
        parser = ConcreteParser(layers, StubConfig())
        result = parser.parse()
        assert isinstance(result, IRModel)
        # input layer + 1 Dense
        assert len(result.layers) == 2
        assert result.layers[1].layer_type == LayerType.DENSE

    def test_parse_skips_non_snn_layers(self):
        layers = [
            {'type': 'Dense', 'output_shape': (None, 10),
             'weights': (np.ones((784, 10), 'f'), np.zeros(10, 'f')),
             'inbound_refs': []},
            {'type': 'Softmax', 'output_shape': (None, 10),
             'inbound_refs': []},
        ]
        parser = ConcreteParser(layers, StubConfig())
        result = parser.parse()
        # Only Dense is in snn_layers
        assert len(result.layers) == 2  # input + Dense

    def test_parse_logs_to_logger(self, caplog):
        layers = [
            {'type': 'Dense', 'output_shape': (None, 10),
             'weights': (np.ones((784, 10), 'f'), np.zeros(10, 'f')),
             'inbound_refs': []},
        ]
        with caplog.at_level(logging.INFO, logger='snntoolbox.parsing'):
            parser = ConcreteParser(layers, StubConfig())
            parser.parse()
        assert any('Parsing layer Dense' in r.message for r in caplog.records)
        assert any('Parsing complete' in r.message for r in caplog.records)

    def test_parse_fires_telemetry(self):
        hook = MagicMock(spec=NullTelemetry)
        layers = [
            {'type': 'Dense', 'output_shape': (None, 10),
             'weights': (np.ones((784, 10), 'f'), np.zeros(10, 'f')),
             'inbound_refs': []},
        ]
        parser = ConcreteParser(layers, StubConfig(), telemetry=hook)
        parser.parse()

        # on_phase_start('parse', ...) was called
        start_calls = [
            c for c in hook.on_phase_start.call_args_list
            if c.args[0] == 'parse'
        ]
        assert len(start_calls) == 1

        # on_phase_end('parse', ...) was called
        end_calls = [
            c for c in hook.on_phase_end.call_args_list
            if c.args[0] == 'parse'
        ]
        assert len(end_calls) == 1
        assert end_calls[0].kwargs['elapsed_sec'] > 0

        # on_metric was called
        metric_names = [c.args[0] for c in hook.on_metric.call_args_list]
        assert 'parse/num_layers' in metric_names
        assert 'parse/elapsed_sec' in metric_names

    def test_layers_to_skip_property(self):
        parser = ConcreteParser([], StubConfig())
        skip = parser.layers_to_skip
        assert 'BatchNormalization' in skip
        assert 'Dropout' in skip
        assert 'Dense' not in skip

    def test_evaluate_raises_by_default(self):
        parser = ConcreteParser([], StubConfig())
        with pytest.raises(NotImplementedError):
            parser.evaluate(None, 1, 1)


# -----------------------------------------------------------------------
# Tests: SNNBackendBase
# -----------------------------------------------------------------------

class TestSNNBackendBase:

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError, match='abstract method'):
            SNNBackendBase(StubConfig())

    def test_build_sets_ir_model(self):
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 784)),
                IRLayer(name='dense_1', layer_type=LayerType.DENSE,
                        output_shape=(None, 10), inbound=('input',),
                        weights=LayerWeights(
                            kernel=np.ones((784, 10), 'f'),
                            bias=np.zeros(10, 'f'))),
            ],
            input_shape=(784,),
        )
        backend = ConcreteBackend(StubConfig())
        backend.build(ir)

        assert backend.is_built is True
        assert backend.ir_model is ir
        assert backend.num_classes == 10

    def test_build_dispatches_layer_types(self):
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 28, 28, 1)),
                IRLayer(name='conv', layer_type=LayerType.CONV2D,
                        output_shape=(None, 26, 26, 32),
                        inbound=('input',)),
                IRLayer(name='pool', layer_type=LayerType.AVERAGE_POOLING_2D,
                        output_shape=(None, 13, 13, 32),
                        inbound=('conv',)),
                IRLayer(name='flat', layer_type=LayerType.FLATTEN,
                        output_shape=(None, 5408),
                        inbound=('pool',)),
                IRLayer(name='dense', layer_type=LayerType.DENSE,
                        output_shape=(None, 10),
                        inbound=('flat',),
                        weights=LayerWeights(
                            kernel=np.ones((5408, 10), 'f'),
                            bias=np.zeros(10, 'f'))),
            ],
            input_shape=(28, 28, 1),
        )
        backend = ConcreteBackend(StubConfig())
        backend.build(ir)

        assert 'input:(1, 28, 28, 1)' in backend._layers_built
        assert 'conv:conv' in backend._layers_built
        assert 'pool:pool' in backend._layers_built
        assert 'dense:dense' in backend._layers_built
        assert 'compile' in backend._layers_built

    def test_build_fires_telemetry(self):
        hook = MagicMock(spec=NullTelemetry)
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 10)),
                IRLayer(name='dense', layer_type=LayerType.DENSE,
                        output_shape=(None, 5), inbound=('input',),
                        weights=LayerWeights(
                            kernel=np.ones((10, 5), 'f'),
                            bias=np.zeros(5, 'f'))),
            ],
            input_shape=(10,),
        )
        backend = ConcreteBackend(StubConfig(), telemetry=hook)
        backend.build(ir)

        start_calls = [
            c for c in hook.on_phase_start.call_args_list
            if c.args[0] == 'build'
        ]
        assert len(start_calls) == 1

        end_calls = [
            c for c in hook.on_phase_end.call_args_list
            if c.args[0] == 'build'
        ]
        assert len(end_calls) == 1
        assert end_calls[0].kwargs['elapsed_sec'] > 0

    def test_build_logs(self, caplog):
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 10)),
                IRLayer(name='dense', layer_type=LayerType.DENSE,
                        output_shape=(None, 5), inbound=('input',)),
            ],
            input_shape=(10,),
        )
        with caplog.at_level(logging.INFO, logger='snntoolbox.simulation'):
            backend = ConcreteBackend(StubConfig())
            backend.build(ir)
        assert any('Building spiking model' in r.message
                    for r in caplog.records)
        assert any('Build complete' in r.message for r in caplog.records)

    def test_run_returns_accuracy(self):
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 4)),
                IRLayer(name='dense', layer_type=LayerType.DENSE,
                        output_shape=(None, 3), inbound=('input',),
                        weights=LayerWeights(
                            kernel=np.ones((4, 3), 'f'),
                            bias=np.zeros(3, 'f'))),
            ],
            input_shape=(4,),
        )
        config = StubConfig({
            ('simulation', 'num_to_test'): '2',
            ('simulation', 'batch_size'): '1',
        })
        backend = ConcreteBackend(config)
        backend.build(ir)

        x = np.random.randn(2, 4).astype('float32')
        y = np.eye(3)[:2]  # classes 0 and 1
        acc = backend.run(x_test=x, y_test=y)

        # ConcreteBackend.simulate always returns correct output
        assert acc == 1.0

    def test_run_fires_batch_telemetry(self):
        hook = MagicMock(spec=NullTelemetry)
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 4)),
                IRLayer(name='dense', layer_type=LayerType.DENSE,
                        output_shape=(None, 3), inbound=('input',)),
            ],
            input_shape=(4,),
        )
        config = StubConfig({
            ('simulation', 'num_to_test'): '2',
            ('simulation', 'batch_size'): '1',
        })
        backend = ConcreteBackend(config, telemetry=hook)
        backend.build(ir)

        hook.reset_mock()
        x = np.random.randn(2, 4).astype('float32')
        y = np.eye(3)[:2]
        backend.run(x_test=x, y_test=y)

        # Check simulate_batch telemetry
        batch_starts = [
            c for c in hook.on_phase_start.call_args_list
            if c.args[0] == 'simulate_batch'
        ]
        assert len(batch_starts) == 2  # 2 batches of 1

        # Check final accuracy metric
        metric_names = [c.args[0] for c in hook.on_metric.call_args_list]
        assert 'run/final_accuracy' in metric_names

    def test_non_parallelizable_adjusts_batchsize(self):
        config = StubConfig({
            ('simulation', 'batch_size'): '8',
        })

        class NonParallel(ConcreteBackend):
            @property
            def is_parallelizable(self):
                return False

        backend = NonParallel(config)
        assert backend.batch_size == 1
