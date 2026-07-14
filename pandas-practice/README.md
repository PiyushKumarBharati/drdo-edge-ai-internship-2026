# pandas-practice

## Purpose

Core pandas workflow — load, inspect, clean, aggregate, and prep features —
practiced on a real dataset (scikit-learn's California Housing, 20,640 rows,
9 columns, downloaded via `sklearn.datasets.fetch_california_housing`), not
invented rows.

## Files

| File | Description |
|---|---|
| `01_load_inspect.py` | `read`/load, `.head()`, `.info()`, `.describe()`, `.dtypes` on the raw dataset. |
| `02_cleaning.py` | Injects 50 synthetic NaNs into `AveBedrms` (clearly marked as synthetic corruption of real data), fills with median, converts `HouseAge` to `int`, renames columns to snake_case. |
| `03_groupby_aggregate.py` | Bins `HouseAge` into brackets, groups by bracket and by a lat/lon grid, multiple aggregations with `.agg()`. |
| `04_feature_prep.py` | One-hot encodes an income-tier categorical, train/test split, exports the cleaned+encoded dataset to CSV. |

## Key concepts covered

- Inspecting an unfamiliar dataset before touching it (shape, dtypes, nulls,
  summary stats).
- The missing-value workflow: detect → decide a fill strategy → verify it
  worked.
- `groupby` + `.agg()` for multi-statistic summaries, and `pd.cut()` for
  turning continuous columns into meaningful buckets.
- One-hot encoding categoricals and keeping a clean train/test split boundary
  (fit only on train, never leak test data into any fitted transform).

## How to run

```bash
python pandas-practice/01_load_inspect.py
python pandas-practice/02_cleaning.py
python pandas-practice/03_groupby_aggregate.py
python pandas-practice/04_feature_prep.py
```

`04_feature_prep.py` writes `california_housing_clean.csv` (20,640 rows, 13
columns, ~2.3 MB) into this folder — that file is committed to the repo.

## Real results from running this code

Mean house value by age bracket (`03_groupby_aggregate.py`, values are
`MedHouseVal` in units of $100k):

| Age bracket | Mean | Median | Count |
|---|---|---|---|
| 0-15 | 1.922 | 1.649 | 3,287 |
| 16-30 | 2.022 | 1.805 | 7,858 |
| 31-45 | 2.062 | 1.782 | 7,284 |
| 46-60 | 2.474 | 2.250 | 2,211 |

Older housing blocks (46-60 years) skew toward *higher* median value in this
dataset — likely reflecting established, desirable neighborhoods rather than
age itself causing value.

Income tier distribution (`04_feature_prep.py`): mid=9,834, low=4,805,
high=4,348, very_high=1,653. Train/test split: 16,512 / 4,128 rows (80/20).

## Why this matters for Edge AI

Before any model — classical or deep — sees data, someone has to answer "is
this clean, is it balanced, what does it actually look like." A feature
pipeline built carelessly on the desktop (wrong dtype, leaked test data,
unbalanced classes) produces a model that looks fine in a notebook and fails
in the field. `04_feature_prep.py`'s train/test discipline is the same
discipline the final CNN in `tensorflow-basics/` and `final-project/` relies
on — evaluate only on data the model never touched during training or feature
fitting.

## Common mistakes / gotchas

- Calling `StandardScaler`/`OneHotEncoder`/etc. `.fit_transform()` on the
  *whole* dataset before splitting leaks test-set statistics into training —
  always split first, or fit only on the train partition.
- `pd.cut()` bin edges are exclusive on the left, inclusive on the right by
  default — an off-by-one in the bin edges silently drops or misclassifies
  boundary rows.
- `df.groupby(..., observed=True)` was used deliberately — pandas'
  categorical groupby defaults can otherwise silently include empty
  categories in the output.
- Injecting missing values with `rng.choice(..., replace=False)` and a fixed
  seed makes `02_cleaning.py` reproducible — without a seed, "missing 50
  rows" would be a different 50 rows every run, making before/after
  comparisons unreliable to discuss.
