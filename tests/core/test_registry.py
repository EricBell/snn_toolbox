"""Tests for the plugin registry."""

import pytest

from snntoolbox.core.registry import PluginRegistry, parser_registry


class TestPluginRegistry:

    def test_explicit_register_and_get(self):
        reg = PluginRegistry(entry_point_group='test.group')
        sentinel = object()
        reg.register('my_plugin', sentinel)
        assert reg.get('my_plugin') is sentinel

    def test_get_missing_raises_key_error(self):
        reg = PluginRegistry(entry_point_group='test.nonexistent')
        with pytest.raises(KeyError, match='No plugin'):
            reg.get('does_not_exist')

    def test_available_includes_registered(self):
        reg = PluginRegistry(entry_point_group='test.group')
        reg.register('alpha', object())
        reg.register('beta', object())
        assert reg.available() == ['alpha', 'beta']

    def test_contains(self):
        reg = PluginRegistry(entry_point_group='test.group')
        reg.register('exists', object())
        assert 'exists' in reg
        assert 'missing' not in reg

    def test_legacy_fallback_import(self):
        reg = PluginRegistry(
            entry_point_group='test.nonexistent',
            legacy_import_template='snntoolbox.core.{name}',
        )
        # Should be able to find 'ir' via legacy import
        module = reg.get('ir')
        assert hasattr(module, 'IRModel')

    def test_legacy_fallback_caches(self):
        reg = PluginRegistry(
            entry_point_group='test.nonexistent',
            legacy_import_template='snntoolbox.core.{name}',
        )
        first = reg.get('ir')
        second = reg.get('ir')
        assert first is second


class TestParserRegistry:

    def test_legacy_keras_parser_discoverable(self):
        """The legacy keras parser should be discoverable via fallback."""
        try:
            module = parser_registry.get('keras')
            assert hasattr(module, 'ModelParser')
            assert hasattr(module, 'load')
        except (ImportError, KeyError):
            pytest.skip('tensorflow not installed')
