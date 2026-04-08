"""Tests for the Intermediate Representation dataclasses."""

import numpy as np
import pytest

from snntoolbox.core.ir import (
    DataFormat,
    IRLayer,
    IRModel,
    LayerType,
    LayerWeights,
    LAYER_TYPE_FROM_STRING,
    LAYER_TYPE_TO_STRING,
)


class TestLayerWeights:

    def test_as_tuple_without_mask(self):
        w = LayerWeights(kernel=np.ones((3, 3)), bias=np.zeros(3))
        t = w.as_tuple()
        assert len(t) == 2
        np.testing.assert_array_equal(t[0], np.ones((3, 3)))
        np.testing.assert_array_equal(t[1], np.zeros(3))

    def test_as_tuple_with_mask(self):
        mask = np.array([1, 0, 1])
        w = LayerWeights(
            kernel=np.ones((3, 3)),
            bias=np.zeros(3),
            mask=mask,
        )
        t = w.as_tuple()
        assert len(t) == 3
        np.testing.assert_array_equal(t[2], mask)

    def test_frozen(self):
        w = LayerWeights(kernel=np.ones(2), bias=np.zeros(2))
        with pytest.raises(AttributeError):
            w.kernel = np.zeros(2)


class TestIRLayer:

    @pytest.fixture()
    def dense_layer(self):
        return IRLayer(
            name='dense_1',
            layer_type=LayerType.DENSE,
            output_shape=(None, 128),
            weights=LayerWeights(
                kernel=np.random.randn(64, 128).astype('float32'),
                bias=np.zeros(128, dtype='float32'),
            ),
            activation='relu',
            inbound=('input',),
        )

    @pytest.fixture()
    def conv_layer(self):
        return IRLayer(
            name='conv2d_1',
            layer_type=LayerType.CONV2D,
            output_shape=(None, 32, 26, 26),
            weights=LayerWeights(
                kernel=np.random.randn(32, 1, 3, 3).astype('float32'),
                bias=np.zeros(32, dtype='float32'),
            ),
            activation='relu',
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            filters=32,
            data_format=DataFormat.CHANNELS_FIRST,
            inbound=('input',),
        )

    def test_has_weights(self, dense_layer):
        assert dense_layer.has_weights is True

    def test_no_weights(self):
        layer = IRLayer(
            name='flatten_1',
            layer_type=LayerType.FLATTEN,
            output_shape=(None, 128),
        )
        assert layer.has_weights is False

    def test_num_neurons(self, dense_layer):
        assert dense_layer.num_neurons == 128

    def test_num_neurons_conv(self, conv_layer):
        assert conv_layer.num_neurons == 32 * 26 * 26

    def test_type_string(self, dense_layer, conv_layer):
        assert dense_layer.type_string == 'Dense'
        assert conv_layer.type_string == 'Conv2D'

    def test_frozen(self, dense_layer):
        with pytest.raises(AttributeError):
            dense_layer.name = 'new_name'

    def test_extra_config_input_shape(self):
        layer = IRLayer(
            name='test',
            layer_type=LayerType.DENSE,
            output_shape=(None, 10),
            extra_config={'input_shape': (None, 128)},
        )
        assert layer.input_shape == (None, 128)

    def test_input_shape_default_none(self):
        layer = IRLayer(
            name='test',
            layer_type=LayerType.DENSE,
            output_shape=(None, 10),
        )
        assert layer.input_shape is None


class TestIRModel:

    @pytest.fixture()
    def simple_model(self):
        return IRModel(
            layers=[
                IRLayer(
                    name='input',
                    layer_type=LayerType.INPUT,
                    output_shape=(None, 28, 28, 1),
                ),
                IRLayer(
                    name='conv2d_1',
                    layer_type=LayerType.CONV2D,
                    output_shape=(None, 26, 26, 32),
                    inbound=('input',),
                    weights=LayerWeights(
                        kernel=np.random.randn(3, 3, 1, 32).astype('float32'),
                        bias=np.zeros(32, dtype='float32'),
                    ),
                    activation='relu',
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    filters=32,
                ),
                IRLayer(
                    name='flatten_1',
                    layer_type=LayerType.FLATTEN,
                    output_shape=(None, 21632),
                    inbound=('conv2d_1',),
                ),
                IRLayer(
                    name='dense_1',
                    layer_type=LayerType.DENSE,
                    output_shape=(None, 10),
                    inbound=('flatten_1',),
                    weights=LayerWeights(
                        kernel=np.random.randn(21632, 10).astype('float32'),
                        bias=np.zeros(10, dtype='float32'),
                    ),
                    activation='softmax',
                ),
            ],
            input_shape=(28, 28, 1),
        )

    def test_get_layer(self, simple_model):
        layer = simple_model.get_layer('conv2d_1')
        assert layer is not None
        assert layer.layer_type == LayerType.CONV2D

    def test_get_layer_missing(self, simple_model):
        assert simple_model.get_layer('nonexistent') is None

    def test_input_output_layers(self, simple_model):
        assert simple_model.input_layer.name == 'input'
        assert simple_model.output_layer.name == 'dense_1'

    def test_num_classes(self, simple_model):
        assert simple_model.num_classes == 10

    def test_layers_with_weights(self, simple_model):
        weighted = simple_model.layers_with_weights()
        assert len(weighted) == 2
        names = {l.name for l in weighted}
        assert names == {'conv2d_1', 'dense_1'}

    def test_get_inbound_layers(self, simple_model):
        dense = simple_model.get_layer('dense_1')
        inbound = simple_model.get_inbound_layers(dense)
        assert len(inbound) == 1
        assert inbound[0].name == 'flatten_1'


class TestLayerTypeMappings:

    def test_all_types_have_string_mapping(self):
        for lt in LayerType:
            assert lt in LAYER_TYPE_TO_STRING

    def test_round_trip(self):
        for string_name, lt in LAYER_TYPE_FROM_STRING.items():
            assert LAYER_TYPE_TO_STRING[lt] == string_name
