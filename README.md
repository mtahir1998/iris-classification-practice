# Iris Flower Classification - Practice 2

**Author**: Muhammad Tahir
**Course**: Software Development Oriented to Machine Learning
**Task**: Project Structure, Training, and Visualization
**Dataset**: Iris (Fisher, 1936) — 150 samples, 3 species, 4 features
**Model**: Logistic Regression (scikit-learn)

## 📖 Documentation website

**https://mtahir1998.github.io/iris-classification-practice/**

The full API reference is published there. It is rebuilt from the docstrings in
`src/` and redeployed automatically by GitHub Actions on every push to `main`,
so it always matches the code in this repository.

## What this project does

This project trains a Logistic Regression classifier to predict the species of
an Iris flower from 4 measurements: sepal length, sepal width, petal length,
and petal width. It covers data exploration, model training, and performance
analysis.

## Design: why the code is split this way

The package is organised so that **each concern lives in exactly one place**:

| File | Single responsibility |
| --- | --- |
| `src/dataset.py` | `Dataset` — loads and preprocesses the data, produces the train/test split |
| `src/model.py` | `Model` — defines the estimator, a thin wrapper around `LogisticRegression` |
| `src/train.py` | One experiment: takes the data and the model, performs the training, saves the result |
| `notebooks/analysis.ipynb` | Loads the trained model and the test data, generates the analysis figures |

The point of this layout is **reusable experiments**. To rerun the last
experiment with a small parameter change, copy `src/train.py`, edit the
constants at the top, and run the copy:

```bash
cp src/train.py src/train_maxiter500.py   # then set MAX_ITER = 500
uv run python -m src.train_maxiter500
```

Each experiment writes to its own folder under `models/`, so runs never
overwrite each other. Because the data and model logic live in `dataset.py`
and `model.py`, a fix made there is inherited by **every** experiment without
editing a single training script.

Visualization is kept out of `src/` entirely. The notebook only reads
artefacts from disk, so analysis can be redone at any time without retraining.

## Project Structure

```
.
├── src/
│   ├── __init__.py           # Package docstring
│   ├── dataset.py            # Dataset class — loading & preprocessing
│   ├── model.py              # Model class — the estimator
│   └── train.py              # Training experiment
├── notebooks/
│   ├── exploration.ipynb     # Data exploration (before training)
│   └── analysis.ipynb        # Model analysis and figures (after training)
├── reports/
│   ├── figures/              # Plots used by the report
│   └── report.md             # Final report (source)
├── pyproject.toml            # Project metadata & dependencies (uv)
├── uv.lock                   # Locked dependency versions
├── LICENSE                   # MIT license
└── README.md                 # This file
```

### Generated directories (not tracked in Git)

These are produced by running the code, so they are deliberately excluded by
`.gitignore` rather than committed. Clone the repository, follow the setup
below, and they rebuild themselves:

| Directory | Produced by | Contents |
| --- | --- | --- |
| `data/` | `uv run python -m src.train` | `raw/iris.csv` and the train/test splits |
| `models/` | `uv run python -m src.train` | One folder per experiment: `model.joblib` + `metadata.json` |
| `docs/` | `uv run pdoc ...` (and CI on every push) | The HTML API documentation |

The dataset itself is never committed — see [Data Source](#data-source) for
where it comes from.

## Setup (dependency management with uv)

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency
management. All dependencies and their version constraints are declared in
`pyproject.toml`, and exact versions are locked in `uv.lock` for full
reproducibility.

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

On a fresh system, one command sets up the entire environment:

```bash
uv sync --all-extras
```

This creates a virtual environment and installs everything (scikit-learn,
pandas, numpy, matplotlib, seaborn, jupyter, pdoc).

> **Why uv?** It handles dependency resolution and virtual-environment
> creation in one step, and produces a lock file (`uv.lock`) that pins exact
> versions — so anyone who clones this repo gets the same environment. Major
> versions are frozen but minor updates are allowed
> (e.g. `scikit-learn>=1.0,<2.0`) unless there is a specific compatibility
> reason. Dependencies needed only for development (pdoc) are declared
> separately under `[project.optional-dependencies]`.

## How to Run

### Step 1: Train the model

```bash
uv run python -m src.train
```

This will:
- Load the Iris dataset and save it to `data/raw/iris.csv`
- Split the data into 80% train / 20% test (stratified, seeded)
- Save the splits to `data/processed/`
- Train a Logistic Regression model
- Save the model to `models/logistic_regression/model.joblib`
- Save parameters and accuracies to `models/logistic_regression/metadata.json`

### Step 2: Generate the analysis figures

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

The notebook loads the saved model and the test split, then produces and saves
to `reports/figures/`:
- Classification report (precision / recall / F1 per species)
- Confusion matrix
- Per-class accuracy bar chart
- Calibration curves

### Step 3: Explore the data (optional)

```bash
uv run jupyter notebook notebooks/exploration.ipynb
```

Interactive exploration with histograms, scatter plots, box plots, and a
correlation heatmap.

## Documentation

The published documentation lives at
**https://mtahir1998.github.io/iris-classification-practice/**.

All modules, classes and methods use **Google-style docstrings** (Args,
Returns, Raises, Attributes, Example). HTML documentation is generated with
**pdoc**.

### Automatic publishing

`.github/workflows/docs.yml` rebuilds the documentation and redeploys it to
GitHub Pages on every push to `main`. Because the site is generated in CI from
the docstrings, the `docs/` folder is not committed — there is no way for the
published documentation to fall out of date with the code.

### Building the documentation locally

To preview it on your own machine before pushing:

```bash
uv run pdoc --docformat google --output-dir docs src
```

> The `--docformat google` flag is required. Without it pdoc assumes
> reStructuredText and renders the `Args:` and `Returns:` sections as a single
> unformatted line.

Then open `docs/index.html` in a browser to browse the API reference.

## Contributing

Contributions are welcome. The workflow below keeps `main` stable and the
published documentation accurate.

### 1. Set up your environment

```bash
git clone https://github.com/mtahir1998/iris-classification-practice.git
cd iris-classification-practice
uv sync --all-extras
```

### 2. Work on a feature branch, never directly on `main`

```bash
git checkout -b feature/short-description
```

Use a prefix that says what the change is: `feature/` for new functionality,
`fix/` for bug fixes, `docs/` for documentation.

### 3. Follow the existing structure

Each file has a single responsibility, and keeping that boundary is the point
of the layout:

| Change | Where it belongs |
| --- | --- |
| Loading or preprocessing data | `src/dataset.py` |
| The estimator or its hyperparameters | `src/model.py` |
| A new experiment | a **copy** of `src/train.py`, with new constants at the top |
| Plots and analysis | `notebooks/analysis.ipynb` |

Do not add plotting code to `src/`, and do not add data loading to
`src/train.py` — that separation is what lets one fix propagate to every
experiment.

### 4. Document what you write

Every public module, class and method needs a **Google-style docstring** with
`Args`, `Returns`, and `Raises` where they apply. Check how it renders before
opening a pull request:

```bash
uv run pdoc --docformat google --output-dir docs src
```

The `--docformat google` flag is required — without it pdoc assumes
reStructuredText and collapses the sections into unformatted text.

### 5. Verify your change actually runs

```bash
uv run python -m src.train
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

Both must finish without errors before you push.

### 6. Write descriptive commit messages

State what changed and why. `Add stratified split to Dataset.split` is useful;
`update`, `fix`, and `changes` are not.

### 7. Open a pull request

Push your branch and open a pull request against `main`, describing what you
changed and how you tested it. Merges into `main` trigger the documentation
workflow, so the published site updates on its own.

## Data Source

The Iris dataset is built into scikit-learn. It was originally published by
R.A. Fisher in 1936 and is one of the most well-known datasets in machine
learning.

- 150 samples total (50 per species)
- 3 species: setosa, versicolor, virginica
- 4 features: sepal length, sepal width, petal length, petal width (all in cm)
- No missing values

Source: `sklearn.datasets.load_iris()`
Original: Fisher, R.A. (1936). "The use of multiple measurements in taxonomic
problems."

## Results

The Logistic Regression model achieves **96.7% accuracy** on the test set
(97.5% on the training set). Setosa is classified perfectly, while versicolor
and virginica occasionally get confused due to overlapping petal
distributions.
