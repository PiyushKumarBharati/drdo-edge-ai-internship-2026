# python-basics

## Purpose

Core Python patterns used throughout the rest of this repo, isolated into
small standalone scripts so each concept can be understood (and defended) on
its own, before it gets buried inside a training or inference pipeline.

## Files

| File | Description |
|---|---|
| `01_data_structures.py` | Lists, dicts, and sets used the way an ML pipeline actually uses them: a batch of paths, a label map, a set of distinct classes. |
| `02_file_io.py` | Reading/writing CSV and JSON with the standard library, no pandas. Round-trips a fake inference log and a run config. |
| `03_oop_basics.py` | A `Dataset` class with `__len__` and `__getitem__`, plus a `BatchedDataset` subclass, the same shape as `torch.utils.data.Dataset` / `tf.data`. |
| `04_exceptions_logging.py` | `try/except/else/finally` + the `logging` module, structured the way an inference script should handle bad/missing/out-of-range input without crashing. |
| `05_list_comprehensions.py` | List/dict comprehensions vs generator expressions, with a real `sys.getsizeof` memory comparison. |

## Key concepts covered

- Choosing the right built-in container (list vs dict vs set) for the job.
- Reading/writing structured data without a heavy dependency.
- Classes as an interface contract (`__len__`, `__getitem__`) rather than just
  a bag of state.
- Defensive input handling with specific exception types plus logging
  instead of silent failure or a hard crash.
- Comprehensions vs generators as a memory/eagerness trade-off, not just
  syntax sugar.

## How to run

From the repo root, with the virtual environment active:

```bash
python python-basics/01_data_structures.py
python python-basics/02_file_io.py
python python-basics/03_oop_basics.py
python python-basics/04_exceptions_logging.py
python python-basics/05_list_comprehensions.py
```

`02_file_io.py` writes `inference_log.csv` and `run_config.json` into this
folder as output.

## Why this matters for Edge AI

Every later phase is one of these five patterns wearing a costume. A
`tf.data.Dataset` is `03_oop_basics.py`'s `Dataset` class with more
features; a TFLite benchmark script is `04_exceptions_logging.py`'s error
handling around a hardware/IO boundary; and the generator-vs-list memory
gap in `05_list_comprehensions.py` is the exact same trade-off as loading an
entire image dataset into RAM vs streaming it batch by batch on a Raspberry
Pi with 1-4 GB of memory.

## Common mistakes / gotchas

- `csv.DictReader` returns every field as a `str`. Forgetting to cast
  `confidence` back to `float` before comparing it numerically is an easy
  bug (see `02_file_io.py`).
- Forgetting `super().__init__()` in a subclass (`BatchedDataset`) silently
  skips the parent's setup instead of raising an error, easy to miss.
- Logging `logger.error(...)` inside an `except` block is not the same as
  re-raising. If the caller needs to know a batch had failures, the function
  has to return that information explicitly (see `process_batch`'s return
  value).
- A generator expression can only be iterated once. Summing it twice would
  give `0` the second time. The script computes the sum fresh each time
  for that reason.
