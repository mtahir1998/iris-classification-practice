"""Iris flower classification — Practice 2.

A small, modular machine learning project that trains a Logistic
Regression classifier on the Iris dataset. The package is split so that
each concern lives in exactly one place, which means a fix made once is
inherited by every experiment.

Modules:
    dataset: Defines :class:`src.dataset.Dataset`, responsible for
        loading the Iris data, preprocessing it, and producing a
        reproducible train/test split.
    model: Defines :class:`src.model.Model`, a thin wrapper around
        :class:`sklearn.linear_model.LogisticRegression` that also
        handles saving and loading.
    train: One training experiment. It takes the data and the model,
        performs the training, and writes the result to
        ``models/<experiment name>/``.

Analysis and figures deliberately live outside this package, in
``notebooks/analysis.ipynb``, so that the training code stays short and
reusable.

Example:
    Run the default experiment from the project root::

        uv run python -m src.train
"""
