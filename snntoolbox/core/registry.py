"""Plugin registry for model parsers and spiking backends.

Replaces the hard-coded ``import_module()`` calls scattered through
``bin/utils.py`` with a single registry that supports three discovery
mechanisms (checked in order):

1. **Explicit registration** via :meth:`PluginRegistry.register`.
2. **Entry points** declared in ``pyproject.toml`` /  ``setup.cfg``
   (groups ``snntoolbox.parsers`` and ``snntoolbox.backends``).
3. **Legacy dynamic import** using the naming convention that the existing
   codebase already follows (e.g.
   ``snntoolbox.parsing.model_libs.{name}_input_lib``).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar('T')


class PluginRegistry(Generic[T]):
    """Generic registry mapping string names to plugin factories.

    Parameters
    ----------
    entry_point_group : str
        The ``importlib.metadata`` entry-point group to scan
        (e.g. ``'snntoolbox.parsers'``).
    legacy_import_template : str, optional
        A Python import path with a ``{name}`` placeholder, used as a
        last-resort fallback.  For example
        ``'snntoolbox.parsing.model_libs.{name}_input_lib'``.
    """

    def __init__(
        self,
        entry_point_group: str,
        legacy_import_template: Optional[str] = None,
    ) -> None:
        self._entry_point_group = entry_point_group
        self._legacy_import_template = legacy_import_template
        self._registry: dict[str, Any] = {}

    def register(
        self,
        name: str,
        factory: type[T] | Callable[..., T],
    ) -> None:
        """Explicitly register a plugin under *name*."""
        self._registry[name] = factory

    def get(self, name: str) -> Any:
        """Look up the plugin registered under *name*.

        Raises
        ------
        KeyError
            If the name cannot be resolved by any discovery mechanism.
        """

        # 1. Explicit registry
        if name in self._registry:
            return self._registry[name]

        # 2. Entry points
        try:
            from importlib.metadata import entry_points as _ep

            # Python >=3.12 returns a SelectableGroups / EntryPoints object;
            # Python 3.9–3.11 needs the group= keyword.
            try:
                eps = _ep(group=self._entry_point_group)
            except TypeError:
                eps = _ep().get(self._entry_point_group, [])

            for ep in eps:
                if ep.name == name:
                    loaded = ep.load()
                    self._registry[name] = loaded
                    return loaded
        except Exception:
            pass

        # 3. Legacy dynamic import
        if self._legacy_import_template is not None:
            module_path = self._legacy_import_template.format(name=name)
            try:
                module = importlib.import_module(module_path)
                self._registry[name] = module
                return module
            except ImportError:
                pass

        raise KeyError(
            f"No plugin '{name}' found in registry "
            f"'{self._entry_point_group}'. "
            f"Available: {self.available()}"
        )

    def available(self) -> list[str]:
        """Return sorted list of all discoverable plugin names."""
        names: set[str] = set(self._registry.keys())
        try:
            from importlib.metadata import entry_points as _ep

            try:
                eps = _ep(group=self._entry_point_group)
            except TypeError:
                eps = _ep().get(self._entry_point_group, [])
            for ep in eps:
                names.add(ep.name)
        except Exception:
            pass
        return sorted(names)

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except KeyError:
            return False


# -----------------------------------------------------------------------
# Singleton registries
# -----------------------------------------------------------------------

parser_registry: PluginRegistry = PluginRegistry(
    entry_point_group='snntoolbox.parsers',
    legacy_import_template='snntoolbox.parsing.model_libs.{name}_input_lib',
)

backend_registry: PluginRegistry = PluginRegistry(
    entry_point_group='snntoolbox.backends',
    legacy_import_template=(
        'snntoolbox.simulation.target_simulators.{name}_target_sim'
    ),
)
