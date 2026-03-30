# Pipeline Engine

**File:** `pipelines/engine.py`

The pipeline engine is the central coordinator of the entire project. It reads
a YAML configuration file, validates it, and then executes each step in order
by dispatching work to the appropriate *handler* class.

---

## Key Classes

### `PipelineConfig` (dataclass)

A [Python dataclass](https://docs.python.org/3/library/dataclasses.html) that
holds the parsed YAML configuration. All fields map directly to top-level YAML
keys.

```python
@dataclass
class PipelineConfig:
    name: str                           # Human-readable pipeline name
    description: str                    # Free-text description
    metadata: Dict[str, Any]            # author, version, created date, etc.
    data: Dict[str, Any]                # data-level preprocessing config
    steps: list                         # ordered list of step definitions
    ludwig: Dict[str, Any]              # Ludwig input/output/trainer config
    sklearn_model: Dict[str, Any]       # sklearn model type + hyperparameters
    experiment: Optional[Dict]          # MLflow experiment config (optional)
    artifacts: Optional[Dict]           # artifact storage config (optional)
    deployment: Optional[Dict]          # deployment config (optional)
```

> **What is a dataclass?**  
> It is a Python class whose fields and types are declared up-front. Python
> automatically generates `__init__`, `__repr__`, and other boilerplate for you.
> Think of it like a structured record / named tuple with default values.

---

### `LudwigPipelineEngine`

The main orchestrator class. You create one instance per pipeline run, passing
the path to a YAML file.

```python
engine = LudwigPipelineEngine("examples/random-forest-anomaly.yaml")
results = engine.execute()
```

#### Constructor

```python
def __init__(self, config_path: str):
```

| Parameter | Type | Description |
|---|---|---|
| `config_path` | `str` | Path to a YAML pipeline configuration file |

Initialises three instance attributes:

| Attribute | Type | Purpose |
|---|---|---|
| `self.config_path` | `Path` | Resolved path to the YAML file |
| `self.config` | `PipelineConfig \| None` | `None` until `load_config()` is called |
| `self.context` | `dict` | Shared data dictionary, empty at start |
| `self.results` | `dict` | Accumulates step results keyed by step name |

---

#### `load_config() → PipelineConfig`

Reads and parses the YAML file using `yaml.safe_load()` (which is safe against
arbitrary code execution), then constructs a `PipelineConfig` dataclass.

```python
config = engine.load_config()
print(config.name)           # e.g. "DU Cell Anomaly Detection with Random Forest"
print(config.sklearn_model)  # {'type': 'random_forest', 'params': {...}, ...}
print(len(config.steps))     # e.g. 6
```

Missing optional keys (like `experiment` or `deployment`) default to `None` or
an empty dict via `config_dict.get(key, {})`.

---

#### `validate_config()`

Inspects the `steps` list to decide which validators to apply.

```
if any step type starts with "ludwig_":
    ├─ config.ludwig must not be empty
    ├─ config.ludwig["input_features"] must exist
    └─ config.ludwig["output_features"] must exist

if any step type starts with "sklearn_":
    ├─ config.sklearn_model must not be empty
    └─ config.sklearn_model["type"] must exist
```

This means a **pure sklearn pipeline** is never asked for a Ludwig config block,
and vice versa. Mixed pipelines must supply both blocks.

Raises `ValueError` with a descriptive message on failure.

---

#### `execute() → Dict[str, Any]`

The main entry point. Calls `load_config()` if not already done, then
`validate_config()`, then iterates over `config.steps`.

For each step it:
1. Extracts `name` and `type` from the step dict.
2. Calls `_execute_step(step)` which returns a result dict.
3. Stores `{status: "success", result: ...}` in `self.results[step_name]`.
4. On exception, stores `{status: "failed", error: str(e)}` and **re-raises**
   the exception (the pipeline stops on first failure).

Returns `self.results` — a dict mapping every step name to its status/result.

```python
results = engine.execute()
# {
#   "load-data": {"status": "success", "result": {"data": df, "shape": (15000, 8)}},
#   "train-sklearn": {"status": "success", "result": {"model": ..., ...}},
#   ...
# }
```

---

#### `_execute_step(step: dict) → Any`

Private method. Looks at `step["type"]` and **dynamically imports** the correct
handler class, instantiates it, calls `.execute()`, then stores the result in
`self.context[step["name"]]`.

**Dynamic import** means the import statement only runs when that handler is
actually needed. This keeps startup time fast and avoids ImportErrors for
libraries that aren't installed (e.g. if Ludwig is not installed but you are
only running sklearn pipelines).

Supported step types and the handler each maps to:

| `type` value | Handler class | Module |
|---|---|---|
| `data_loader` | `DataLoaderHandler` | `pipelines.handlers.data_loader` |
| `unsupervised_labeling` | `UnsupervisedLabelingHandler` | `pipelines.handlers.data_loader` |
| `ludwig_train` | `LudwigTrainHandler` | `pipelines.handlers.ludwig_handler` |
| `ludwig_evaluate` | `LudwigEvaluateHandler` | `pipelines.handlers.ludwig_handler` |
| `ludwig_visualize` | `LudwigVisualizeHandler` | `pipelines.handlers.ludwig_visualize` |
| `sklearn_train` | `SklearnTrainHandler` | `pipelines.handlers.sklearn_handler` |
| `sklearn_evaluate` | `SklearnEvaluateHandler` | `pipelines.handlers.sklearn_handler` |
| `sklearn_visualize` | `SklearnVisualizeHandler` | `pipelines.handlers.sklearn_handler` |
| `sklearn_xai` | `SklearnXAIHandler` | `pipelines.handlers.sklearn_handler` |

Any other value raises `ValueError: Unknown step type: <value>`.

---

#### `main()` (module-level function)

A thin CLI wrapper that reads `sys.argv[1]` as the config path and calls
`engine.execute()`. Used when running the module directly:

```bash
python -m pipelines.engine examples/random-forest-anomaly.yaml
```

The preferred way to run pipelines is via `run_pipeline.py`, which uses argparse
and provides `--output` and `--gpu` flags.

---

## Logging

The engine uses Python's standard `logging` module configured at `INFO` level
with timestamps:

```
2026-02-07 12:00:01,234 - pipelines.engine - INFO - Initializing Ludwig Pipeline Engine
2026-02-07 12:00:01,235 - pipelines.engine - INFO - Config: examples/random-forest-anomaly.yaml
```

Each handler also has its own logger (`logger = logging.getLogger(__name__)`),
so you can see which module is producing each line.

To increase verbosity to `DEBUG`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## How to Add a New Step Type

1. Create a handler class in an appropriate module under `pipelines/handlers/`:

    ```python
    class MyNewHandler:
        def __init__(self, config, context):
            self.config = config
            self.context = context

        def execute(self):
            # ... do work ...
            return {"my_output": "value"}
    ```

2. Add an `elif` branch in `LudwigPipelineEngine._execute_step()`:

    ```python
    elif step_type == 'my_new_step':
        from pipelines.handlers.my_module import MyNewHandler
        handler = MyNewHandler(step_config, self.context)
    ```

3. Use the new step in any YAML pipeline:

    ```yaml
    steps:
      - name: my-step
        type: my_new_step
        config:
          my_param: some_value
    ```

---

## Error Handling

The engine follows a **fail-fast** strategy: the first step that raises an
exception causes `execute()` to re-raise immediately. The error is logged at
`ERROR` level and the `results` dict records `status: "failed"`.

There is no automatic retry logic. For production use, wrap `engine.execute()`
in your own retry loop if needed.

---

## Testing the Engine

See `tests/test_engine.py` for unit tests covering:
- Config loading
- Config validation (sklearn vs Ludwig)
- Unknown step type errors

To run just the engine tests:

```bash
pytest tests/test_engine.py -v
```
