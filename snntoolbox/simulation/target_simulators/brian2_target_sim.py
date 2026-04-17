"""Building and simulating spiking neural networks using Brian2.

Implements the :class:`SNNBackendBase` interface for the Brian2 simulator.
"""

import logging
import os
import warnings
from typing import Any, Optional

import numpy as np

from snntoolbox.bin.utils import get_log_keys, get_plot_keys
from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
    LAYER_TYPE_FROM_STRING,
)
from snntoolbox.core.protocols import SNNBackendBase, TelemetryHook
from snntoolbox.simulation.utils import (
    build_convolution,
    build_pooling,
    get_shape_from_label,
)
from snntoolbox.utils.utils import confirm_overwrite


class SNN(SNNBackendBase):
    """Spiking neural network compiled for the Brian2 simulator.

    Attributes
    ----------
    layers : list
        Brian2 ``NeuronGroup`` objects, one per network layer.
    connections : list
        Brian2 ``Synapses`` objects connecting consecutive layers.
    spikemonitors : list
        Brian2 ``SpikeMonitor`` objects recording spikes per layer.
    statemonitors : list
        Brian2 ``StateMonitor`` objects recording membrane potentials.
    snn : brian2.Network
        The compiled Brian2 network.
    """

    def __init__(
        self,
        config,
        *,
        queue: Any = None,
        logger: Optional[logging.Logger] = None,
        telemetry: Optional[TelemetryHook] = None,
    ):
        self._parallelizable = False
        super().__init__(config, queue=queue, logger=logger, telemetry=telemetry)

        import brian2
        self.sim = brian2

        self.layers: list = []
        self.connections: list = []
        self.spikemonitors: list = []
        self.statemonitors: list = []
        self.snn = None
        self._input_layer = None
        self._cell_params = None
        self.output_spikemonitor = None
        self.flatten_shapes: list = []
        self.data_format = None

        self.threshold = 'v >= v_thresh'
        if 'subtraction' in config.get('cell', 'reset'):
            self.v_reset = 'v = v - v_thresh'
        else:
            self.v_reset = 'v = v_reset'
        self.eqs = '''dv/dt = bias : 1
                      bias : hertz'''

        self.rescale_fac = 1000 / (
            config.getint('input', 'input_rate') * self._dt
        )
        self._poisson_input = config.getboolean('input', 'poisson_input')
        self._plot_keys = get_plot_keys(config)
        self._log_keys = get_log_keys(config)
        self._spiketrains_container_counter = None

    @property
    def is_parallelizable(self):
        return False

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def add_input_layer(self, input_shape):
        if self._poisson_input:
            self.layers.append(self.sim.PoissonGroup(
                np.prod(input_shape[1:]), rates=0 * self.sim.Hz,
                dt=self._dt * self.sim.ms))
        else:
            self.layers.append(self.sim.NeuronGroup(
                np.prod(input_shape[1:]), model=self.eqs, method='euler',
                reset=self.v_reset, threshold=self.threshold,
                dt=self._dt * self.sim.ms))
        self.layers[0].add_attribute('label')
        self.layers[0].label = 'InputLayer'
        self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[0]))
        self.statemonitors.append(
            self.sim.StateMonitor(self.layers[0], [], False))

    def add_layer(self, layer: IRLayer):
        if layer.layer_type == LayerType.FLATTEN:
            self.flatten_shapes.append(
                (layer.name, get_shape_from_label(self.layers[-1].label)))
            return

        self.layers.append(self.sim.NeuronGroup(
            layer.num_neurons, model=self.eqs, method='euler',
            reset=self.v_reset, threshold=self.threshold,
            dt=self._dt * self.sim.ms))
        self.connections.append(self.sim.Synapses(
            self.layers[-2], self.layers[-1], 'w:1', on_pre='v+=w',
            dt=self._dt * self.sim.ms))
        self.layers[-1].add_attribute('label')
        self.layers[-1].label = layer.name

        if 'spiketrains' in self._plot_keys \
                or 'spiketrains_n_b_l_t' in self._log_keys:
            self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        if 'v_mem' in self._plot_keys or 'mem_n_b_l_t' in self._log_keys:
            self.statemonitors.append(
                self.sim.StateMonitor(self.layers[-1], 'v', True))

    def build_dense(self, layer: IRLayer):
        if layer.activation == 'softmax':
            warnings.warn(
                "Activation 'softmax' not implemented. "
                "Using 'relu' activation instead.",
                RuntimeWarning,
                stacklevel=2,
            )

        weights, biases = layer.weights.kernel, layer.weights.bias
        self._set_biases(biases)

        delay = self.config.getfloat('cell', 'delay')

        if len(self.flatten_shapes) == 1:
            self.logger.debug('Swapping data_format of Flatten layer.')
            flatten_name, shape = self.flatten_shapes.pop()
            if self.data_format == 'channels_last':
                y_in, x_in, f_in = shape
            else:
                f_in, y_in, x_in = shape
            src = np.arange(weights.shape[0])
            f = src % f_in
            y = src // (f_in * x_in)
            x = (src // f_in) % x_in
            remapped = f * x_in * y_in + x_in * y + x
            ii, jj = np.meshgrid(remapped, np.arange(weights.shape[1]),
                                 indexing='ij')
            connections = np.column_stack([
                ii.ravel(),
                jj.ravel(),
                weights.ravel(),
                np.full(weights.size, delay),
            ])
        elif len(self.flatten_shapes) > 1:
            raise RuntimeError(
                "Not all Flatten layers have been consumed.")
        else:
            ii, jj = np.indices(weights.shape)
            connections = np.column_stack([
                ii.ravel(),
                jj.ravel(),
                weights.ravel(),
                np.full(weights.size, delay),
            ])

        self.connections[-1].connect(
            i=connections[:, 0].astype('int64'),
            j=connections[:, 1].astype('int64'),
        )
        self.connections[-1].w = connections[:, 2]

    def build_convolution(self, layer: IRLayer):
        delay = self.config.getfloat('cell', 'delay')
        transpose_kernel = \
            self.config.get('simulation', 'keras_backend') == 'tensorflow'
        conns, biases = build_convolution(layer, delay, transpose_kernel)
        connections = np.array(conns)

        self._set_biases(biases)

        self.logger.debug('Connecting layer...')
        self.connections[-1].connect(
            i=connections[:, 0].astype('int64'),
            j=connections[:, 1].astype('int64'),
        )
        self.connections[-1].w = connections[:, 2]

    def build_pooling(self, layer: IRLayer):
        if layer.layer_type == LayerType.MAX_POOLING_2D:
            warnings.warn(
                "Layer type 'MaxPooling' not supported yet. "
                "Falling back on 'AveragePooling'.",
                RuntimeWarning,
                stacklevel=2,
            )

        delay = self.config.getfloat('cell', 'delay')
        connections = np.array(build_pooling(layer, delay))

        self.connections[-1].connect(
            i=connections[:, 0].astype('int64'),
            j=connections[:, 1].astype('int64'),
        )
        self.connections[-1].w = connections[:, 2]

    # ------------------------------------------------------------------
    # Compile / simulate / reset
    # ------------------------------------------------------------------

    def compile(self):
        self.output_spikemonitor = self.sim.SpikeMonitor(self.layers[-1])
        spikemonitors = self.spikemonitors + [self.output_spikemonitor]
        self.snn = self.sim.Network(
            self.layers, self.connections, spikemonitors, self.statemonitors)
        self.snn.store()

        for obj in self.snn.objects:
            if hasattr(obj, 'label') and obj.label == 'InputLayer':
                self._input_layer = obj
        assert self._input_layer, 'No input layer found.'

    def simulate(self, **kwargs):
        inputs = kwargs['x_b_l'].flatten() / self.sim.ms
        if self._poisson_input:
            self._input_layer.rates = inputs / self.rescale_fac
        else:
            self._input_layer.bias = inputs

        self.snn.run(
            self._duration * self.sim.ms,
            namespace=self._cell_params,
            report='stdout',
            report_period=10 * self.sim.ms,
        )

        return self._assemble_output()

    def reset(self, sample_idx):
        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        if sample_idx % mod == 0:
            self.logger.debug('Resetting simulator...')
            self.snn.restore()

    def end_sim(self):
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path, filename):
        self.logger.info('Saving weights...')
        for i, connection in enumerate(self.connections):
            filepath = os.path.join(
                path,
                self.config.get('paths', 'filename_snn'),
                'brian2-model',
                self.layers[i + 1].label + '.npz',
            )
            if self.config.getboolean('output', 'overwrite') \
                    or confirm_overwrite(filepath):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                self.logger.info(
                    'Storing weights of layer %s to %s',
                    self.layers[i + 1].label, filepath,
                )
                np.savez(filepath, self.connections[i].w)

    def load(self, path, filename):
        from tensorflow.keras.models import load_model

        dirpath = os.path.join(path, filename, 'brian2-model')
        npz_files = sorted(
            f for f in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, f))
        )
        self.logger.info('Loading spiking model...')

        parsed_model = load_model(
            os.path.join(
                self.config.get('paths', 'path_wd'),
                self.config.get('paths', 'filename_parsed_model') + '.h5',
            )
        )

        ir_layers = [
            self._keras_layer_to_ir(keras_layer, npz_path=os.path.join(dirpath, f))
            for keras_layer, f in zip(parsed_model.layers[1:], npz_files)
        ]

        ir_model = IRModel(
            layers=ir_layers,
            input_shape=tuple(parsed_model.input_shape[1:]),
            data_format=(
                DataFormat.CHANNELS_FIRST
                if hasattr(parsed_model.layers[1], 'data_format')
                and parsed_model.layers[1].data_format == 'channels_first'
                else DataFormat.CHANNELS_LAST
            ),
        )

        self.build(ir_model)

    @staticmethod
    def _keras_layer_to_ir(keras_layer, *, npz_path: str) -> IRLayer:
        """Convert a Keras layer + saved .npz weights into an IRLayer."""
        from snntoolbox.parsing.utils import get_type

        layer_type_str = get_type(keras_layer)
        layer_type = LAYER_TYPE_FROM_STRING.get(layer_type_str, LayerType.DENSE)

        data = np.load(npz_path)
        kernel = data['arr_0']
        bias = np.zeros(kernel.shape[-1]) if kernel.ndim >= 2 else np.zeros(1)
        weights = LayerWeights(kernel=kernel, bias=bias)

        extra = {}
        if hasattr(keras_layer, 'input_shape'):
            extra['input_shape'] = keras_layer.input_shape

        kwargs = {
            'name': keras_layer.name,
            'layer_type': layer_type,
            'output_shape': keras_layer.output_shape,
            'weights': weights,
            'extra_config': extra,
        }

        if hasattr(keras_layer, 'kernel_size'):
            kwargs['kernel_size'] = keras_layer.kernel_size
        if hasattr(keras_layer, 'strides'):
            kwargs['strides'] = keras_layer.strides
        if hasattr(keras_layer, 'padding'):
            kwargs['padding'] = keras_layer.padding
        if hasattr(keras_layer, 'filters'):
            kwargs['filters'] = keras_layer.filters
        if hasattr(keras_layer, 'pool_size'):
            kwargs['pool_size'] = keras_layer.pool_size
        if hasattr(keras_layer, 'data_format'):
            fmt = keras_layer.data_format
            kwargs['data_format'] = (
                DataFormat.CHANNELS_FIRST if fmt == 'channels_first'
                else DataFormat.CHANNELS_LAST
            )

        return IRLayer(**kwargs)

    # ------------------------------------------------------------------
    # Cell initialization
    # ------------------------------------------------------------------

    def init_cells(self):
        self._cell_params = {
            'v_thresh': self.config.getfloat('cell', 'v_thresh'),
            'v_reset': self.config.getfloat('cell', 'v_reset'),
            'tau_m': self.config.getfloat('cell', 'tau_m') * self.sim.ms,
        }

    # ------------------------------------------------------------------
    # Recording accessors
    # ------------------------------------------------------------------

    def get_spiketrains(self, **kwargs):
        monitor_index = kwargs['monitor_index']
        i = (len(self.spikemonitors) - 1
             if monitor_index == -1
             else monitor_index + 1)
        if i >= len(self.spikemonitors):
            return None
        spiketrain_dict = self.spikemonitors[i].spike_trains()
        return np.array([
            spiketrain_dict[key] / self.sim.ms for key in spiketrain_dict
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_output(self):
        """Build cumulative spike-count output from the output spike monitor."""
        shape = (self.batch_size, self.num_classes, self._num_timesteps)
        output_b_l_t = np.zeros(shape, 'int32')
        spiketrain_dict = self.output_spikemonitor.spike_trains()

        for neuron_id, spike_times in spiketrain_dict.items():
            times_ms = np.asarray(spike_times / self.sim.ms)
            timesteps = (times_ms / self._dt).astype(int)
            valid = (timesteps >= 0) & (timesteps < self._num_timesteps)
            for ts in timesteps[valid]:
                output_b_l_t[0, int(neuron_id), ts:] += 1

        return output_b_l_t

    def _set_biases(self, biases):
        if np.any(biases):
            assert self.layers[-1].bias.shape == biases.shape, \
                'Shape of biases and network do not match.'
            self.layers[-1].bias = biases / self.sim.ms
