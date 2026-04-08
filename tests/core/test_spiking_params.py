"""Tests for the pure-Python spiking neuron parameter library.

Every test in this module runs without TensorFlow or any other ML framework.
"""

from collections import OrderedDict

import numpy as np
import pytest

from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
)
from snntoolbox.core.spiking_params import (
    absorb_bn_parameters,
    apply_normalization_schedule,
    binarize,
    compute_scale_factors,
    get_percentile,
    get_scale_fac,
    modify_parameter_precision,
    normalize_weights,
    reduce_precision,
)


# -----------------------------------------------------------------------
# Stub config for testing
# -----------------------------------------------------------------------

class StubConfig:
    def __init__(self, overrides=None):
        self._d = {
            ('normalization', 'percentile'): '99.9',
            ('normalization', 'normalization_schedule'): 'false',
            ('cell', 'binarize_weights'): 'false',
            ('cell', 'quantize_weights'): 'false',
        }
        if overrides:
            self._d.update(overrides)

    def get(self, s, k):
        return self._d.get((s, k), '')

    def getint(self, s, k):
        return int(self._d[(s, k)])

    def getfloat(self, s, k):
        return float(self._d[(s, k)])

    def getboolean(self, s, k):
        return self._d[(s, k)].lower() in ('true', '1', 'yes')

    def set(self, s, k, v):
        self._d[(s, k)] = v


# -----------------------------------------------------------------------
# absorb_bn_parameters
# -----------------------------------------------------------------------

class TestAbsorbBN:

    def test_identity_bn(self):
        """BN with gamma=1, beta=0, mean=0, var=1 should be identity."""
        w = np.random.randn(3, 3, 1, 8).astype('float32')
        b = np.random.randn(8).astype('float32')
        n = 8
        w_bn, b_bn = absorb_bn_parameters(
            w, b,
            mean=np.zeros(n),
            var_eps_sqrt_inv=np.ones(n),
            gamma=np.ones(n),
            beta=np.zeros(n),
            axis=-1,
            image_data_format='channels_last',
        )
        np.testing.assert_allclose(w_bn, w, atol=1e-6)
        np.testing.assert_allclose(b_bn, b, atol=1e-6)

    def test_nontrivial_bn(self):
        """Non-trivial BN parameters should modify weights."""
        w = np.ones((3, 3, 1, 4), dtype='float32')
        b = np.zeros(4, dtype='float32')
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0])
        mean = np.zeros(4)
        var_inv = np.ones(4)

        w_bn, b_bn = absorb_bn_parameters(
            w, b, mean, var_inv, gamma, beta,
            axis=-1, image_data_format='channels_last',
        )
        # weight should be scaled by gamma * var_inv = 2
        np.testing.assert_allclose(w_bn, 2.0)
        # bias_bn = beta + (bias - mean) * gamma * var_inv = 1 + 0 = 1
        np.testing.assert_allclose(b_bn, 1.0)

    def test_channels_first(self):
        """channels_first should remap axis correctly."""
        w = np.ones((3, 3, 1, 4), dtype='float32')
        b = np.zeros(4, dtype='float32')
        n = 4
        w_bn, b_bn = absorb_bn_parameters(
            w, b,
            mean=np.zeros(n), var_eps_sqrt_inv=np.ones(n),
            gamma=np.ones(n) * 3, beta=np.zeros(n),
            axis=1, image_data_format='channels_first',
        )
        np.testing.assert_allclose(w_bn, 3.0)

    def test_dense_2d(self):
        """Dense (2D weight) should work without axis remapping."""
        w = np.ones((10, 5), dtype='float32')
        b = np.zeros(5, dtype='float32')
        n = 5
        w_bn, b_bn = absorb_bn_parameters(
            w, b,
            mean=np.zeros(n), var_eps_sqrt_inv=np.ones(n),
            gamma=np.ones(n) * 2, beta=np.zeros(n),
            axis=-1, image_data_format='channels_last',
        )
        np.testing.assert_allclose(w_bn, 2.0)

    def test_depthwise(self):
        """Depthwise conv uses channel_axis=2."""
        w = np.ones((3, 3, 4, 1), dtype='float32')
        b = np.zeros(4, dtype='float32')
        n = 4
        w_bn, _ = absorb_bn_parameters(
            w, b,
            mean=np.zeros(n), var_eps_sqrt_inv=np.ones(n),
            gamma=np.ones(n) * 5, beta=np.zeros(n),
            axis=-1, image_data_format='channels_last',
            is_depthwise=True,
        )
        np.testing.assert_allclose(w_bn, 5.0)


# -----------------------------------------------------------------------
# binarize / reduce_precision
# -----------------------------------------------------------------------

class TestBinarize:

    def test_positive_values(self):
        w = np.array([0.5, 1.0, 0.9], dtype='float32')
        wb = binarize(w)
        np.testing.assert_array_equal(wb, [1.0, 1.0, 1.0])

    def test_negative_values(self):
        w = np.array([-0.5, -1.0, -0.9], dtype='float32')
        wb = binarize(w)
        np.testing.assert_array_equal(wb, [-1.0, -1.0, -1.0])

    def test_custom_h(self):
        w = np.array([0.5, -0.5], dtype='float32')
        wb = binarize(w, h=2.0)
        assert np.all(np.isin(wb, [-2.0, 2.0]))

    def test_output_dtype(self):
        wb = binarize(np.array([0.1]))
        assert wb.dtype == np.float32


class TestReducePrecision:

    def test_basic(self):
        x = np.array([1.234, -0.567])
        x_lp = reduce_precision(x, m=2, f=4)
        # Should be rounded to Q2.4 grid
        assert x_lp.shape == x.shape
        # Values should be clipped within Q2.4 range
        n = 2 << 3  # 16
        maxval = (2 << 1) - 1.0 / n
        assert np.all(x_lp <= maxval)
        assert np.all(x_lp >= -maxval)

    def test_identity_high_precision(self):
        x = np.array([0.5, -0.5])
        # Very high precision should preserve values closely
        x_lp = reduce_precision(x, m=8, f=16)
        np.testing.assert_allclose(x_lp, x, atol=1e-4)


# -----------------------------------------------------------------------
# modify_parameter_precision
# -----------------------------------------------------------------------

class TestModifyParameterPrecision:

    def test_no_modification(self):
        w = np.ones((3, 3))
        b = np.zeros(3)
        config = StubConfig()
        attrs = {}
        w_out, b_out = modify_parameter_precision(w, b, config, attrs)
        np.testing.assert_array_equal(w_out, w)
        np.testing.assert_array_equal(b_out, b)

    def test_binarize(self):
        w = np.random.randn(5, 5).astype('float32')
        b = np.zeros(5, dtype='float32')
        config = StubConfig({('cell', 'binarize_weights'): 'true'})
        attrs = {}
        w_out, _ = modify_parameter_precision(w, b, config, attrs)
        assert np.all(np.isin(w_out, [-1.0, 1.0]))

    def test_quantize(self):
        w = np.random.randn(5, 5).astype('float32')
        b = np.zeros(5, dtype='float32')
        config = StubConfig({('cell', 'quantize_weights'): 'true'})
        attrs = {'Qm.f': (2, 4)}
        w_out, _ = modify_parameter_precision(w, b, config, attrs)
        assert 'Qm.f' not in attrs  # Cleaned up


# -----------------------------------------------------------------------
# Scale factor computation
# -----------------------------------------------------------------------

class TestGetScaleFac:

    def test_basic_percentile(self):
        acts = np.arange(100, dtype='float32')
        sf = get_scale_fac(acts, 99.0)
        expected = float(np.percentile(acts, 99.0))
        assert sf == pytest.approx(expected, abs=0.01)

    def test_empty_returns_one(self):
        sf = get_scale_fac(np.array([]), 99.0)
        assert sf == 1.0

    def test_single_value(self):
        sf = get_scale_fac(np.array([5.0]), 99.0)
        assert sf == pytest.approx(5.0)


class TestGetPercentile:

    def test_default(self):
        config = StubConfig()
        assert get_percentile(config) == pytest.approx(99.9)

    def test_with_schedule(self):
        config = StubConfig({
            ('normalization', 'normalization_schedule'): 'true',
        })
        p = get_percentile(config, layer_idx=5)
        expected = 99.9 - 5 * 0.02
        assert p == pytest.approx(expected)


class TestApplyNormalizationSchedule:

    def test_basic(self):
        assert apply_normalization_schedule(99.9, 0) == pytest.approx(99.9)
        assert apply_normalization_schedule(99.9, 10) == pytest.approx(99.7)


# -----------------------------------------------------------------------
# IR-level normalization
# -----------------------------------------------------------------------

def _make_two_layer_model(
    w1=None, b1=None, w2=None, b2=None,
):
    """Helper: input → dense_1 → dense_2 (output)."""
    if w1 is None:
        w1 = np.ones((4, 8), dtype='float32')
    if b1 is None:
        b1 = np.ones(8, dtype='float32')
    if w2 is None:
        w2 = np.ones((8, 3), dtype='float32')
    if b2 is None:
        b2 = np.ones(3, dtype='float32')
    return IRModel(
        layers=[
            IRLayer(name='input', layer_type=LayerType.INPUT,
                    output_shape=(None, 4)),
            IRLayer(name='dense_1', layer_type=LayerType.DENSE,
                    output_shape=(None, 8), inbound=('input',),
                    activation='relu',
                    weights=LayerWeights(kernel=w1, bias=b1)),
            IRLayer(name='dense_2', layer_type=LayerType.DENSE,
                    output_shape=(None, 3), inbound=('dense_1',),
                    activation='softmax',
                    weights=LayerWeights(kernel=w2, bias=b2)),
        ],
        input_shape=(4,),
    )


class TestComputeScaleFactors:

    def test_basic(self):
        ir = _make_two_layer_model()
        activations = {
            'dense_1': np.arange(1, 101, dtype='float32'),
            'dense_2': np.arange(1, 51, dtype='float32'),
        }
        config = StubConfig()
        sf = compute_scale_factors(ir, activations, config)
        assert 'input' in sf
        assert sf['input'] == 1.0
        assert 'dense_1' in sf
        assert sf['dense_1'] > 0
        assert 'dense_2' in sf

    def test_missing_activations_uses_one(self):
        ir = _make_two_layer_model()
        sf = compute_scale_factors(ir, {}, StubConfig())
        assert sf.get('dense_1') == 1.0


class TestNormalizeWeights:

    def test_single_inbound(self):
        ir = _make_two_layer_model()
        scale_facs = {'input': 1.0, 'dense_1': 2.0, 'dense_2': 4.0}
        norm = normalize_weights(ir, scale_facs)

        # dense_1: kernel * input_sf / layer_sf = 1 * 1 / 2 = 0.5
        np.testing.assert_allclose(
            norm.get_layer('dense_1').weights.kernel, 0.5,
        )
        # dense_1: bias / layer_sf = 1 / 2 = 0.5
        np.testing.assert_allclose(
            norm.get_layer('dense_1').weights.bias, 0.5,
        )

    def test_softmax_layer_uses_scale_one(self):
        ir = _make_two_layer_model()
        scale_facs = {'input': 1.0, 'dense_1': 2.0, 'dense_2': 4.0}
        norm = normalize_weights(ir, scale_facs)

        # dense_2 has activation='softmax', so scale_fac=1.0
        # kernel * inbound_sf / 1.0 = 1 * 2 / 1 = 2
        np.testing.assert_allclose(
            norm.get_layer('dense_2').weights.kernel, 2.0,
        )
        # bias / 1.0 = 1.0
        np.testing.assert_allclose(
            norm.get_layer('dense_2').weights.bias, 1.0,
        )

    def test_does_not_mutate_original(self):
        w1 = np.ones((4, 8), dtype='float32')
        ir = _make_two_layer_model(w1=w1)
        scale_facs = {'input': 1.0, 'dense_1': 2.0, 'dense_2': 1.0}
        normalize_weights(ir, scale_facs)
        # Original weights should be unchanged
        np.testing.assert_array_equal(
            ir.get_layer('dense_1').weights.kernel, 1.0,
        )

    def test_preserves_sparse_mask(self):
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype='float32')
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 4)),
                IRLayer(name='sparse_1', layer_type=LayerType.SPARSE,
                        output_shape=(None, 8), inbound=('input',),
                        activation='relu',
                        weights=LayerWeights(
                            kernel=np.ones((4, 8), 'f'),
                            bias=np.zeros(8, 'f'),
                            mask=mask)),
            ],
            input_shape=(4,),
        )
        scale_facs = {'input': 1.0, 'sparse_1': 2.0}
        norm = normalize_weights(ir, scale_facs)
        np.testing.assert_array_equal(
            norm.get_layer('sparse_1').weights.mask, mask,
        )

    def test_layer_without_weights_passes_through(self):
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 28, 28, 1)),
                IRLayer(name='flatten', layer_type=LayerType.FLATTEN,
                        output_shape=(None, 784), inbound=('input',)),
                IRLayer(name='dense', layer_type=LayerType.DENSE,
                        output_shape=(None, 10), inbound=('flatten',),
                        activation='relu',
                        weights=LayerWeights(
                            kernel=np.ones((784, 10), 'f'),
                            bias=np.zeros(10, 'f'))),
            ],
            input_shape=(28, 28, 1),
        )
        scale_facs = {'input': 1.0, 'dense': 2.0}
        norm = normalize_weights(ir, scale_facs)
        assert norm.get_layer('flatten').weights is None
        assert norm.get_layer('dense').weights is not None

    def test_shapes_preserved(self):
        ir = _make_two_layer_model()
        scale_facs = {'input': 1.0, 'dense_1': 2.0, 'dense_2': 3.0}
        norm = normalize_weights(ir, scale_facs)
        assert len(norm.layers) == len(ir.layers)
        for orig, new in zip(ir.layers, norm.layers):
            assert orig.output_shape == new.output_shape
            assert orig.name == new.name
            if orig.has_weights:
                assert new.weights.kernel.shape == orig.weights.kernel.shape
                assert new.weights.bias.shape == orig.weights.bias.shape


class TestNormalizeWeightsMultiInput:

    def test_conv_after_concatenate(self):
        """Conv layer receiving concatenated input from two conv layers."""
        ir = IRModel(
            layers=[
                IRLayer(name='input', layer_type=LayerType.INPUT,
                        output_shape=(None, 8, 8, 1)),
                IRLayer(name='conv_a', layer_type=LayerType.CONV2D,
                        output_shape=(None, 8, 8, 4), inbound=('input',),
                        filters=4, activation='relu',
                        weights=LayerWeights(
                            kernel=np.ones((3, 3, 1, 4), 'f'),
                            bias=np.zeros(4, 'f'))),
                IRLayer(name='conv_b', layer_type=LayerType.CONV2D,
                        output_shape=(None, 8, 8, 4), inbound=('input',),
                        filters=4, activation='relu',
                        weights=LayerWeights(
                            kernel=np.ones((3, 3, 1, 4), 'f'),
                            bias=np.zeros(4, 'f'))),
                IRLayer(name='concat', layer_type=LayerType.CONCATENATE,
                        output_shape=(None, 8, 8, 8),
                        inbound=('conv_a', 'conv_b')),
                IRLayer(name='conv_c', layer_type=LayerType.CONV2D,
                        output_shape=(None, 8, 8, 16),
                        inbound=('concat',), filters=16,
                        activation='relu',
                        weights=LayerWeights(
                            kernel=np.ones((1, 1, 8, 16), 'f'),
                            bias=np.zeros(16, 'f'))),
            ],
            input_shape=(8, 8, 1),
        )
        scale_facs = {
            'input': 1.0,
            'conv_a': 2.0,
            'conv_b': 4.0,
            'conv_c': 8.0,
        }
        norm = normalize_weights(ir, scale_facs)
        k = norm.get_layer('conv_c').weights.kernel

        # First 4 input channels scaled by conv_a sf / conv_c sf = 2/8
        np.testing.assert_allclose(k[:, :, :4, :], 2.0 / 8.0)
        # Last 4 input channels scaled by conv_b sf / conv_c sf = 4/8
        np.testing.assert_allclose(k[:, :, 4:, :], 4.0 / 8.0)
