"""Tests for the pure-Python connectivity library.

Every test in this module runs without TensorFlow or any other ML framework.
"""

import numpy as np
import pytest

from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
)
from snntoolbox.core.connectivity import (
    ConnectivityStats,
    compute_ann_ops,
    compute_connectivity,
    get_fanin,
    get_fanout,
    get_fanout_array,
    get_spiking_outbound_layers,
    has_stride_unity,
    is_conv,
    is_pool,
    is_spiking,
    num_neurons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input_layer(shape=(28, 28, 1), name='input'):
    return IRLayer(
        name=name,
        layer_type=LayerType.INPUT,
        output_shape=(None,) + tuple(shape),
    )


def _conv2d(
    name,
    in_shape,
    filters=8,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='valid',
    inbound=('input',),
    data_format=DataFormat.CHANNELS_LAST,
):
    h, w, c = in_shape
    kx, ky = kernel_size
    sx, sy = strides
    if padding == 'valid':
        out_h = (h - ky) // sy + 1
        out_w = (w - kx) // sx + 1
    else:  # 'same'
        out_h = -(-h // sy)
        out_w = -(-w // sx)
    return IRLayer(
        name=name,
        layer_type=LayerType.CONV2D,
        output_shape=(None, out_h, out_w, filters),
        inbound=tuple(inbound),
        weights=LayerWeights(
            kernel=np.zeros((ky, kx, c, filters), dtype=np.float32),
            bias=np.zeros(filters, dtype=np.float32),
        ),
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        filters=filters,
        data_format=data_format,
        extra_config={'input_shape': (None, h, w, c)},
    )


def _dense(name, in_features, units, inbound, has_bias_values=False):
    bias = (
        np.ones(units, dtype=np.float32)
        if has_bias_values
        else np.zeros(units, dtype=np.float32)
    )
    return IRLayer(
        name=name,
        layer_type=LayerType.DENSE,
        output_shape=(None, units),
        inbound=tuple(inbound),
        weights=LayerWeights(
            kernel=np.zeros((in_features, units), dtype=np.float32),
            bias=bias,
        ),
        extra_config={'input_shape': (None, in_features)},
    )


def _pool2d(name, in_shape, pool_size=(2, 2), strides=(2, 2), inbound=()):
    h, w, c = in_shape
    return IRLayer(
        name=name,
        layer_type=LayerType.AVERAGE_POOLING_2D,
        output_shape=(None, h // strides[0], w // strides[1], c),
        inbound=tuple(inbound),
        pool_size=pool_size,
        strides=strides,
        padding='valid',
    )


def _flatten(name, in_shape, inbound):
    total = int(np.prod(in_shape))
    return IRLayer(
        name=name,
        layer_type=LayerType.FLATTEN,
        output_shape=(None, total),
        inbound=tuple(inbound),
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class TestPredicates:

    def test_is_spiking_dense_and_conv(self):
        assert is_spiking(_dense('d', 10, 5, inbound=('input',)))
        assert is_spiking(_conv2d('c', (8, 8, 4)))

    def test_is_spiking_false_for_flatten(self):
        assert not is_spiking(_flatten('f', (2, 2, 4), inbound=('c',)))

    def test_is_spiking_false_for_input(self):
        assert not is_spiking(_input_layer())

    def test_is_conv_and_pool(self):
        assert is_conv(_conv2d('c', (8, 8, 4)))
        assert not is_pool(_conv2d('c', (8, 8, 4)))
        assert is_pool(_pool2d('p', (8, 8, 4), inbound=('c',)))

    def test_stride_unity(self):
        assert has_stride_unity(_conv2d('c', (8, 8, 4), strides=(1, 1)))
        assert not has_stride_unity(_conv2d('c', (8, 8, 4), strides=(2, 2)))


# ---------------------------------------------------------------------------
# num_neurons
# ---------------------------------------------------------------------------


class TestNumNeurons:

    def test_input_layer_uses_shape(self):
        model = IRModel(layers=[_input_layer((28, 28, 1))], input_shape=(28, 28, 1))
        assert num_neurons(model.input_layer, model) == 784

    def test_conv_layer_product_of_output_shape(self):
        conv = _conv2d('c', (28, 28, 1), filters=16, kernel_size=(3, 3))
        # valid padding -> output (26, 26, 16) = 10816
        assert num_neurons(conv) == 26 * 26 * 16


# ---------------------------------------------------------------------------
# Fan-in
# ---------------------------------------------------------------------------


class TestFanin:

    def test_conv_fanin_kernel_times_channels(self):
        conv = _conv2d('c', (28, 28, 3), filters=16, kernel_size=(3, 3))
        # 3x3 * 3 input channels = 27
        assert get_fanin(conv) == 27

    def test_dense_fanin_is_in_features(self):
        dense = _dense('d', 784, 10, inbound=('flat',))
        assert get_fanin(dense) == 784

    def test_pool_fanin_is_zero(self):
        pool = _pool2d('p', (8, 8, 4), inbound=('c',))
        assert get_fanin(pool) == 0

    def test_conv_channels_first(self):
        conv = IRLayer(
            name='c',
            layer_type=LayerType.CONV2D,
            output_shape=(None, 16, 26, 26),
            inbound=('input',),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            filters=16,
            data_format=DataFormat.CHANNELS_FIRST,
            extra_config={'input_shape': (None, 3, 28, 28)},
        )
        # 3x3 * 3 channels = 27
        assert get_fanin(conv) == 27

    def test_fanin_falls_back_to_inbound_layer(self):
        conv = _conv2d('c', (28, 28, 3), filters=8, kernel_size=(3, 3))
        # Drop the cached input_shape and provide the model instead
        conv_no_input = IRLayer(
            name=conv.name,
            layer_type=conv.layer_type,
            output_shape=conv.output_shape,
            inbound=('input',),
            kernel_size=conv.kernel_size,
            strides=conv.strides,
            padding=conv.padding,
            filters=conv.filters,
            data_format=conv.data_format,
        )
        model = IRModel(
            layers=[_input_layer((28, 28, 3)), conv_no_input],
            input_shape=(28, 28, 3),
        )
        assert get_fanin(conv_no_input, model) == 27


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


class TestFanout:

    def test_fanout_into_dense_is_units_count(self):
        inp = _input_layer((28, 28, 1))
        flat = _flatten('f', (28, 28, 1), inbound=('input',))
        dense = _dense('d', 784, 10, inbound=('f',))
        model = IRModel(
            layers=[inp, flat, dense], input_shape=(28, 28, 1),
        )
        # Input-layer fan-out walks past Flatten, lands on Dense, so fanout = 10
        assert get_fanout(inp, model) == 10

    def test_fanout_into_conv_stride1_is_kprod_times_filters(self):
        inp = _input_layer((28, 28, 1))
        conv = _conv2d('c', (28, 28, 1), filters=16, kernel_size=(3, 3))
        model = IRModel(layers=[inp, conv], input_shape=(28, 28, 1))
        # 3*3 * 16 = 144
        assert get_fanout(inp, model) == 144

    def test_fanout_into_conv_stride2_returns_array(self):
        inp = _input_layer((8, 8, 1))
        conv = _conv2d(
            'c', (8, 8, 1), filters=4, kernel_size=(3, 3), strides=(2, 2),
        )
        model = IRModel(layers=[inp, conv], input_shape=(8, 8, 1))
        fo = get_fanout(inp, model)
        assert hasattr(fo, 'shape')
        assert fo.shape == (8, 8, 1)
        # Centre pixel drives 3x3 = 9 post-synaptic neurons per channel
        # (before multiplying by nz); after multiplying by 4 filters, >= 4
        assert np.all(fo >= 1)

    def test_fanout_into_pool_is_one(self):
        inp = _input_layer((8, 8, 4))
        pool = _pool2d('p', (8, 8, 4), inbound=('input',))
        model = IRModel(layers=[inp, pool], input_shape=(8, 8, 4))
        assert get_fanout(inp, model) == 1

    def test_fanout_no_downstream_spiking_layer(self):
        inp = _input_layer((4,))
        flat = _flatten('f', (4,), inbound=('input',))
        model = IRModel(layers=[inp, flat], input_shape=(4,))
        # Flatten is non-spiking and has no outbound -> fanout is 0
        assert get_fanout(inp, model) == 0


# ---------------------------------------------------------------------------
# get_spiking_outbound_layers
# ---------------------------------------------------------------------------


class TestSpikingOutbound:

    def test_walks_past_flatten(self):
        inp = _input_layer((28, 28, 1))
        flat = _flatten('f', (28, 28, 1), inbound=('input',))
        dense = _dense('d', 784, 10, inbound=('f',))
        model = IRModel(
            layers=[inp, flat, dense], input_shape=(28, 28, 1),
        )
        spiking = get_spiking_outbound_layers(inp, model)
        assert [l.name for l in spiking] == ['d']

    def test_branching(self):
        inp = _input_layer((8, 8, 4))
        branch1 = _conv2d('b1', (8, 8, 4), filters=8, inbound=('input',))
        branch2 = _conv2d('b2', (8, 8, 4), filters=8, inbound=('input',))
        model = IRModel(
            layers=[inp, branch1, branch2], input_shape=(8, 8, 4),
        )
        spiking = get_spiking_outbound_layers(inp, model)
        assert {l.name for l in spiking} == {'b1', 'b2'}


# ---------------------------------------------------------------------------
# compute_connectivity
# ---------------------------------------------------------------------------


class TestConnectivityStats:

    def test_simple_dense_network(self):
        inp = _input_layer((4,))
        dense = _dense('d', 4, 2, inbound=('input',), has_bias_values=True)
        model = IRModel(layers=[inp, dense], input_shape=(4,))

        stats = compute_connectivity(model)

        assert stats.num_neurons == [4, 2]
        assert stats.num_neurons_with_bias == [0, 2]
        assert stats.fanin == [0, 4]
        # Each input neuron drives 2 post-synaptic (dense.units); each dense
        # neuron drives 0 downstream -> synapse total = 4*2 + 2*0 = 8
        assert stats.num_synapses == 8

    def test_conv_pool_dense(self):
        inp = _input_layer((8, 8, 1))
        conv = _conv2d(
            'c', (8, 8, 1), filters=4, kernel_size=(3, 3), padding='same',
        )
        pool = _pool2d('p', (8, 8, 4), inbound=('c',))
        flat = _flatten('f', (4, 4, 4), inbound=('p',))
        dense = _dense('d', 64, 10, inbound=('f',))
        model = IRModel(
            layers=[inp, conv, pool, flat, dense], input_shape=(8, 8, 1),
        )
        stats = compute_connectivity(model)

        # num_neurons: input=64, conv=256, pool=64, dense=10 (flatten skipped)
        assert stats.num_neurons == [64, 256, 64, 10]
        # Only spiking layers get bias tracking
        assert stats.num_neurons_with_bias == [0, 0, 0, 0]
        # Dense fan-in is 64; conv fan-in is 3*3*1=9; pool fan-in is 0
        assert stats.fanin == [0, 9, 0, 64]

    def test_bias_predicate_detects_nonzero_bias(self):
        inp = _input_layer((4,))
        dense = _dense('d', 4, 3, inbound=('input',), has_bias_values=True)
        model = IRModel(layers=[inp, dense], input_shape=(4,))

        stats = compute_connectivity(model)
        assert stats.num_neurons_with_bias[-1] == 3


# ---------------------------------------------------------------------------
# compute_ann_ops
# ---------------------------------------------------------------------------


class TestAnnOps:

    def test_matches_legacy_formula(self):
        # Legacy: 2 * dot(num_neurons, fanin) + sum(num_neurons_with_bias)
        num_neurons = [784, 100, 10]
        fanin = [0, 784, 100]
        num_with_bias = [0, 100, 10]
        ops = compute_ann_ops(num_neurons, num_with_bias, fanin)
        expected = 2 * (0 + 784 * 100 + 10 * 100) + 110
        assert ops == expected
