"""Tests for the legacy ↔ IR adapter layer."""

import numpy as np
import pytest

from snntoolbox.core.adapters import (
    IRLayerFacade,
    IRModelFacade,
    layer_list_to_ir,
)
from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
)


class TestLayerListToIR:

    def test_basic_conversion(self):
        layer_list = [
            {
                'layer_type': 'Dense',
                'name': '00Dense_128',
                'inbound': ['input'],
                'parameters': (
                    np.ones((784, 128), dtype='float32'),
                    np.zeros(128, dtype='float32'),
                ),
                'activation': 'relu',
            },
            {
                'layer_type': 'Dense',
                'name': '01Dense_10',
                'inbound': ['00Dense_128'],
                'parameters': (
                    np.ones((128, 10), dtype='float32'),
                    np.zeros(10, dtype='float32'),
                ),
                'activation': 'softmax',
            },
        ]

        ir = layer_list_to_ir(layer_list, input_shape=(784,))

        assert len(ir.layers) == 3  # input + 2 Dense
        assert ir.layers[0].layer_type == LayerType.INPUT
        assert ir.layers[1].name == '00Dense_128'
        assert ir.layers[1].activation == 'relu'
        assert ir.layers[1].has_weights
        assert ir.layers[2].activation == 'softmax'

    def test_conv_layer(self):
        kernel = np.random.randn(3, 3, 1, 32).astype('float32')
        bias = np.zeros(32, dtype='float32')
        layer_list = [
            {
                'layer_type': 'Conv2D',
                'name': '00Conv2D_32',
                'inbound': ['input'],
                'parameters': (kernel, bias),
                'activation': 'relu',
                'kernel_size': (3, 3),
                'strides': (1, 1),
                'padding': 'valid',
                'filters': 32,
                'output_shape': (None, 26, 26, 32),
            },
        ]

        ir = layer_list_to_ir(layer_list, input_shape=(28, 28, 1))

        conv = ir.layers[1]
        assert conv.layer_type == LayerType.CONV2D
        assert conv.kernel_size == (3, 3)
        assert conv.strides == (1, 1)
        assert conv.filters == 32
        np.testing.assert_array_equal(conv.weights.kernel, kernel)

    def test_sparse_layer_with_mask(self):
        kernel = np.ones((10, 5), dtype='float32')
        bias = np.zeros(5, dtype='float32')
        mask = np.array([1, 0, 1, 0, 1])
        layer_list = [
            {
                'layer_type': 'Sparse',
                'name': '00Sparse_5',
                'inbound': ['input'],
                'parameters': (kernel, bias, mask),
                'activation': 'relu',
            },
        ]

        ir = layer_list_to_ir(layer_list, input_shape=(10,))

        sparse = ir.layers[1]
        assert sparse.layer_type == LayerType.SPARSE
        assert sparse.weights.mask is not None
        np.testing.assert_array_equal(sparse.weights.mask, mask)

    def test_skips_unknown_layer_types(self):
        layer_list = [
            {
                'layer_type': 'UnknownLayer',
                'name': 'unknown',
                'inbound': ['input'],
            },
        ]
        ir = layer_list_to_ir(layer_list, input_shape=(10,))
        # Only input layer
        assert len(ir.layers) == 1

    def test_input_shape_preserved(self):
        ir = layer_list_to_ir([], input_shape=(28, 28, 1))
        assert ir.input_shape == (28, 28, 1)


class TestIRLayerFacade:

    @pytest.fixture()
    def facade(self):
        ir_layer = IRLayer(
            name='conv2d_1',
            layer_type=LayerType.CONV2D,
            output_shape=(None, 26, 26, 32),
            weights=LayerWeights(
                kernel=np.ones((3, 3, 1, 32), dtype='float32'),
                bias=np.zeros(32, dtype='float32'),
            ),
            activation='relu',
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            filters=32,
            data_format=DataFormat.CHANNELS_LAST,
            extra_config={'input_shape': (None, 28, 28, 1)},
        )
        return IRLayerFacade(ir_layer)

    def test_name(self, facade):
        assert facade.name == 'conv2d_1'

    def test_class_name(self, facade):
        assert facade.__class__.__name__ == 'Conv2D'

    def test_output_shape(self, facade):
        assert facade.output_shape == (None, 26, 26, 32)

    def test_input_shape(self, facade):
        assert facade.input_shape == (None, 28, 28, 1)

    def test_get_weights(self, facade):
        w = facade.get_weights()
        assert len(w) == 2
        assert w[0].shape == (3, 3, 1, 32)
        assert w[1].shape == (32,)

    def test_no_weights(self):
        layer = IRLayer(
            name='flatten',
            layer_type=LayerType.FLATTEN,
            output_shape=(None, 128),
        )
        facade = IRLayerFacade(layer)
        assert facade.get_weights() == []

    def test_kernel_size(self, facade):
        assert facade.kernel_size == (3, 3)

    def test_strides(self, facade):
        assert facade.strides == (1, 1)

    def test_padding(self, facade):
        assert facade.padding == 'valid'

    def test_filters(self, facade):
        assert facade.filters == 32

    def test_data_format(self, facade):
        assert facade.data_format == 'channels_last'

    def test_activation_name(self, facade):
        assert facade.activation.__name__ == 'relu'

    def test_get_config(self, facade):
        config = facade.get_config()
        assert config['name'] == 'conv2d_1'
        assert config['kernel_size'] == (3, 3)
        assert config['filters'] == 32

    def test_bias(self, facade):
        assert facade.bias is not None
        assert facade.bias.shape == (32,)


class TestIRModelFacade:

    @pytest.fixture()
    def model_facade(self):
        ir = IRModel(
            layers=[
                IRLayer(
                    name='input',
                    layer_type=LayerType.INPUT,
                    output_shape=(None, 28, 28, 1),
                ),
                IRLayer(
                    name='dense_1',
                    layer_type=LayerType.DENSE,
                    output_shape=(None, 10),
                    inbound=('input',),
                    weights=LayerWeights(
                        kernel=np.ones((784, 10), dtype='float32'),
                        bias=np.zeros(10, dtype='float32'),
                    ),
                    activation='softmax',
                ),
            ],
            input_shape=(28, 28, 1),
        )
        return IRModelFacade(ir)

    def test_layers_count(self, model_facade):
        assert len(model_facade.layers) == 2

    def test_layers_are_facades(self, model_facade):
        assert isinstance(model_facade.layers[0], IRLayerFacade)
        assert isinstance(model_facade.layers[1], IRLayerFacade)

    def test_input_shape(self, model_facade):
        assert model_facade.input_shape == (None, 28, 28, 1)

    def test_output_shape(self, model_facade):
        assert model_facade.output_shape == (None, 10)

    def test_layer_names(self, model_facade):
        names = [l.name for l in model_facade.layers]
        assert names == ['input', 'dense_1']

    def test_get_type_compatibility(self, model_facade):
        """Verify that get_type(layer) pattern works with facade."""
        layer = model_facade.layers[1]
        # This is how the legacy code does it:
        layer_type = layer.__class__.__name__
        assert layer_type == 'Dense'
