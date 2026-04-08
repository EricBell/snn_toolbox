"""Protocol definitions for the snn_toolbox plugin system.

These :class:`~typing.Protocol` classes define the contracts that model parsers
and spiking backends must satisfy.  Using structural subtyping (Python 3.8+),
new implementations can satisfy a protocol via duck typing without inheriting
from the legacy abstract base classes.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from snntoolbox.core.ir import IRModel


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
class ModelParser(Protocol):
    """Protocol for parsing an ANN into the framework-agnostic IR.

    Implementations load a framework-specific model from disk and produce
    an :class:`~snntoolbox.core.ir.IRModel`.
    """

    def load(
        self,
        path: str,
        filename: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load a model from disk.

        Returns
        -------
        dict
            Must contain at least a ``'model'`` key with the loaded model
            object.
        """
        ...

    def parse(self, input_model: Any, config: ToolboxConfig) -> IRModel:
        """Parse *input_model* into an :class:`~snntoolbox.core.ir.IRModel`.

        Parameters
        ----------
        input_model
            Framework-specific model object (e.g. a Keras ``Model``).
        config
            Toolbox configuration for this experiment.
        """
        ...

    def evaluate(
        self,
        val_fn: Any,
        batch_size: int,
        num_to_test: int,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        dataflow: Any = None,
    ) -> float:
        """Evaluate model accuracy on the given test data."""
        ...


@runtime_checkable
class SpikingBackend(Protocol):
    """Protocol for spiking neural network simulators.

    Implementations consume an :class:`~snntoolbox.core.ir.IRModel` (or a
    facade wrapping one) and simulate the resulting spiking network.
    """

    @property
    def is_parallelizable(self) -> bool:
        """Whether the simulator can test multiple samples in parallel."""
        ...

    def build(self, ir_model: IRModel, **kwargs: Any) -> None:
        """Build the spiking network from the IR."""
        ...

    def run(
        self,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        dataflow: Any = None,
        **kwargs: Any,
    ) -> float:
        """Simulate and return top-1 accuracy."""
        ...

    def simulate(self, **kwargs: Any) -> np.ndarray:
        """Simulate one batch; return ``output_b_l_t`` array."""
        ...

    def reset(self, sample_idx: int) -> None:
        """Reset network state between samples."""
        ...

    def end_sim(self) -> None:
        """Clean up after simulation."""
        ...

    def save(self, path: str, filename: str) -> None:
        """Serialize the spiking model to disk."""
        ...

    def load(self, path: str, filename: str) -> None:
        """Restore a spiking model from disk."""
        ...


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
