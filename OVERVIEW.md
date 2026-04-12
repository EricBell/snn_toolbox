# snntoolbox — Overview

> A Python framework that converts pre-trained rate-based ANNs (Keras, PyTorch, Caffe, Lasagne) into spiking neural networks and runs or deploys them on simulation backends (INI, Brian2, PyNN) or neuromorphic hardware (Intel Loihi, SpiNNaker).

## Purpose

SNN-TB (Spiking Neural Network Toolbox) takes an already-trained analog neural network from any supported deep-learning framework, normalizes its weights, and translates it into a spiking network. The converted network can be simulated locally with several spike-coding strategies (temporal mean rate, TTFS, temporal pattern) or deployed to neuromorphic hardware. The toolbox is research-oriented: it exposes every stage of the conversion pipeline as a configurable step, supports parameter sweeps, and produces spike-rate / voltage plots for analysis.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9–3.12 |
| Package / env management | uv + hatchling |
| Core deps | numpy, h5py, matplotlib |
| ML framework (optional) | TensorFlow ≥ 2.10 (`[tensorflow]`), PyTorch + ONNX (`[pytorch]`) |
| SNN simulation (optional) | Brian2 (`[brian2]`), PyNN (`[pynn]`) |
| Hardware targets | Intel Loihi (proprietary nxsdk, not on PyPI), SpiNNaker, MegaSim |
| Event-camera data | more-itertools (`[aedat]`) |
| Test framework | pytest, pytest-cov |
| Docs | Sphinx + sphinx-rtd-theme |

## Directory Structure

```
snntoolbox/
├── bin/
│   ├── run.py              # CLI entry point (`snntoolbox` command) and GUI launcher
│   ├── utils.py            # run_pipeline() — the 5-stage orchestration loop
│   └── gui/                # Tkinter GUI
├── core/                   # Framework-agnostic abstractions (added in refactor)
│   ├── ir.py               # IRModel / IRLayer dataclasses (the intermediate representation)
│   ├── protocols.py        # ModelParserBase ABC, SNNBackendBase ABC, TelemetryHook protocol
│   ├── adapters.py         # Keras↔IR bridges: keras_model_to_ir(), IRLayerFacade, IRModelFacade
│   ├── registry.py         # PluginRegistry for parsers and backends
│   └── spiking_params.py   # Pure-numpy weight normalization, BN absorption, scale factors
├── parsing/
│   └── model_libs/
│       ├── keras_input_lib.py      # Keras/TF parser (primary path)
│       ├── pytorch_input_lib.py    # PyTorch parser (converts via ONNX → Keras internally)
│       ├── caffe_input_lib.py
│       └── lasagne_input_lib.py
├── conversion/
│   └── utils.py            # Weight normalization (Keras-side; delegates to spiking_params)
├── simulation/
│   ├── target_simulators/  # One module per backend (INI_*, brian2, loihi, spiNNaker, …)
│   ├── backends/
│   │   ├── inisim/         # Custom TF/Keras spiking layers for INI backends
│   │   └── megasim/
│   ├── utils.py            # AbstractSNN base class (legacy, pre-refactor)
│   └── plotting.py         # Spike-rate / v_mem / confusion-matrix plots
├── datasets/
│   ├── utils.py            # get_dataset(): loads npz, jpg, aedat data
│   └── aedat/              # Event-camera AEDAT format reader
├── utils/
│   └── utils.py            # Misc helpers (import_script, etc.)
└── config_defaults         # Single INI-format file — all config keys and defaults

tests/                      # pytest suite, mirrors package structure
examples/                   # Runnable MNIST / ResNet scripts per backend
docs/                       # Sphinx documentation source
```

## Architecture

### 5-Stage Pipeline (`bin/utils.py:run_pipeline`)

1. **Load dataset** — `datasets/utils.py:get_dataset()` returns a normalization set and a test set (npz arrays or a Keras dataflow).
2. **Parse ANN** — the configured `model_lib` module (e.g. `keras_input_lib`) loads the pre-trained model, a `ModelParser` walks its layers, and `core/adapters.py:layer_list_to_ir()` converts the layer list to a framework-agnostic `IRModel`.
3. **Normalize weights** — `conversion/utils.py:normalize_parameters()` runs forward passes on the normalization set, computes per-layer scale factors via `core/spiking_params.py:compute_scale_factors()`, and produces a new `IRModel` with adjusted weights.
4. **Build SNN** — `import_target_sim(config)` loads the backend module, instantiates `SNN(config)` (which extends `SNNBackendBase` or the legacy `AbstractSNN`), and calls `build(ir_model)`.
5. **Simulate** — `backend.run(x_test, y_test)` runs batched simulation, computing top-1 accuracy and optionally logging spike trains / membrane voltages.

### Intermediate Representation (`core/ir.py`)

`IRModel` / `IRLayer` are frozen dataclasses that carry everything a backend needs: weights (`LayerWeights`), output shape, connectivity (`inbound` names), activation, and layer-type-specific attributes (kernel_size, strides, padding, etc.). This decouples parsers from backends — no framework import is needed to consume an `IRModel`.

`LayerType` is an enum; `LAYER_TYPE_FROM_STRING` maps legacy Keras string names to it.

### Protocol / ABC Layer (`core/protocols.py`)

- `ModelParserBase` (ABC) — template method `parse()` orchestrates `get_layer_iterable()` → per-layer `parse_dense/parse_convolution/…` → `layer_list_to_ir()`. Subclasses implement the framework-specific `get_*` / `parse_*` methods.
- `SNNBackendBase` (ABC) — template methods `build()` and `run()` orchestrate `add_input_layer()` → per-layer `build_dense/build_convolution/…` → `compile()` → batched `simulate()`.
- `TelemetryHook` (Protocol) — `on_phase_start / on_phase_end / on_metric`. The default `NullTelemetry` is a no-op; implementations can log to W&B, MLflow, CSV, etc.

### Plugin Registry (`core/registry.py`)

`PluginRegistry` resolves parser/backend names in priority order:
1. Explicit `register()` call
2. `importlib.metadata` entry points (declared in `pyproject.toml` under `snntoolbox.parsers` / `snntoolbox.backends`)
3. Legacy dynamic import (`snntoolbox.parsing.model_libs.{name}_input_lib`)

Singletons: `parser_registry` and `backend_registry`.

### Backward-Compatibility Adapters (`core/adapters.py`)

- `keras_model_to_ir(keras_model)` — converts an existing Keras `Model` to `IRModel` without touching parser code.
- `IRLayerFacade` / `IRModelFacade` — wrap `IRLayer` / `IRModel` to expose the Keras-layer API (`get_weights()`, `.output_shape`, `.kernel_size`, …) so that legacy `AbstractSNN` backends work transparently with the new IR.

## Integrations

| Integration | Purpose | Location |
|---|---|---|
| TensorFlow / Keras | Primary parser and INI backends | `parsing/model_libs/keras_input_lib.py`, `simulation/backends/inisim/` |
| PyTorch | Model source (converted via ONNX → Keras) | `parsing/model_libs/pytorch_input_lib.py` |
| ONNX + onnxruntime | PyTorch → Keras bridge | `parsing/model_libs/pytorch_input_lib.py` |
| Brian2 | SNN simulation backend | `simulation/target_simulators/brian2_target_sim.py` |
| PyNN | Abstraction over nest/brian/neuron/SpiNNaker | `simulation/target_simulators/pyNN_target_sim.py` |
| Intel Loihi (nxsdk) | Neuromorphic hardware deployment | `simulation/target_simulators/loihi_target_sim.py` |
| SpiNNaker | Neuromorphic hardware deployment | `simulation/target_simulators/spiNNaker_target_sim.py` |

## Database & Data Layer

No database. Data is loaded from the filesystem:
- **npz** — NumPy archive with `x_test`, `y_test` arrays
- **jpg** — image directory (Keras `ImageDataGenerator`)
- **aedat** — event-camera recordings (via `datasets/aedat/`)

## Connectivity & Configuration

All configuration is managed through a single INI-format file passed on the CLI (`snntoolbox <config_filepath>`). Defaults live in `snntoolbox/config_defaults`. Key sections and notable keys:

| Section | Key | Purpose |
|---|---|---|
| `paths` | `path_wd`, `filename_ann` | Working directory and model filename |
| `input` | `model_lib` | Parser to use (`keras`, `pytorch`, `caffe`, `lasagne`) |
| `input` | `poisson_input`, `input_rate` | Spike encoding for input layer |
| `tools` | `parse`, `normalize`, `convert`, `simulate` | Toggle pipeline stages on/off |
| `normalization` | `percentile` | Activation percentile for weight scaling (default 99.9) |
| `conversion` | `spike_code` | Encoding: `temporal_mean_rate`, `ttfs`, `temporal_pattern`, etc. |
| `simulation` | `simulator` | Backend: `INI`, `brian2`, `MegaSim`, `loihi`, `nest`, etc. |
| `simulation` | `duration`, `dt`, `batch_size` | Simulation time window and stepping |
| `cell` | `v_thresh`, `tau_refrac`, `reset` | Neuron parameters |
| `parameter_sweep` | `param_name`, `param_values` | Grid search over a single parameter |

No environment variables are used; all paths are file-based.

## Key Entry Points

| File | What to read |
|---|---|
| `snntoolbox/bin/run.py` | CLI parsing and GUI launch |
| `snntoolbox/bin/utils.py:run_pipeline` | End-to-end 5-stage orchestration |
| `snntoolbox/core/ir.py` | `IRModel` / `IRLayer` data structures |
| `snntoolbox/core/protocols.py` | `ModelParserBase` and `SNNBackendBase` ABCs |
| `snntoolbox/core/adapters.py` | Parser↔backend bridge |
| `snntoolbox/core/registry.py` | Plugin discovery |
| `snntoolbox/config_defaults` | All config keys and their defaults |
| `snntoolbox/parsing/model_libs/keras_input_lib.py` | Reference parser implementation |
| `snntoolbox/simulation/target_simulators/INI_temporal_mean_rate_target_sim.py` | Reference backend |

## Notes & Gotchas

- **PyTorch parser goes through ONNX then Keras**: `pytorch_input_lib.py` imports `tensorflow.keras` and converts the PyTorch model to ONNX, then loads it into a Keras model. TensorFlow is therefore a *runtime* dependency of the PyTorch parser — not just an optional extra.
- **`bin/utils.py` still hard-codes a Keras import**: `from snntoolbox.parsing.model_libs.keras_input_lib import load` is at module level. The new `PluginRegistry` in `core/registry.py` is not yet wired into `run_pipeline()`; both paths coexist during the ongoing migration.
- **`core/` is a recent addition**: The IR, protocols, adapters, and registry were introduced in the refactoring commits to decouple parsers from backends. Legacy `AbstractSNN` and legacy parsers still exist in parallel; `IRLayerFacade` / `IRModelFacade` bridge keeps them working.
- **Loihi requires proprietary nxsdk**: Intel's `nxsdk` and `nxsdk_modules_ncl` are not on PyPI. The `loihi` optional-dependency group in `pyproject.toml` is intentionally empty.
- **`config_defaults` is a file, not a directory**: Despite the name, it is a single INI file force-included in the wheel via `hatch.build.targets.wheel.force-include`.
- **Batch size auto-reduced to 1** for non-parallelizable backends: `SNNBackendBase._adjust_batchsize()` checks `is_parallelizable` and silently overrides the config value.
- **`channels_first` required for PyTorch**: The PyTorch ONNX bridge assumes `channels_first` data ordering. Tests set this explicitly in `setup_module` and restore it in `teardown_module`.
- **Use `uv`, not pip**: `uv run pytest`, `uv add <pkg>`, `uv sync --extra dev` are the expected workflows.
