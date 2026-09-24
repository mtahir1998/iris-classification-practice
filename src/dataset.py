"""Dataset loading and preprocessing for the Iris classification project.

This module owns everything related to *data*: reading the raw Iris
measurements, writing them to disk as CSV, and producing a reproducible
train/test split. Training and plotting live elsewhere, so a fix applied
here is inherited by every experiment that uses this class.

Example:
    Prepare the data and inspect the split::

        from src.dataset import Dataset

        dataset = Dataset(test_size=0.2, random_state=42)
        dataset.prepare()
        print(dataset.X_train.shape)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""pathlib.Path: Absolute path to the project root directory."""

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
"""pathlib.Path: Default directory for the untouched source data."""

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
"""pathlib.Path: Default directory for the generated train/test splits."""

LABEL_COLUMN = "species"
"""str: Name of the target column in the processed CSV files."""


class Dataset:
    """Loads, preprocesses and splits the Iris flower dataset.

    The Iris dataset ships with scikit-learn, so nothing is downloaded.
    It contains 150 samples of 3 flower species, each described by 4
    petal and sepal measurements in centimetres.

    A single instance walks through three stages. After ``load()`` the
    full data is available, after ``split()`` the train and test
    attributes are populated, and ``prepare()`` simply runs both.

    Args:
        test_size (float): Fraction of the data held out for testing.
            Defaults to ``0.2`` (20%).
        random_state (int): Seed used for the split, so the same rows
            land in the same set on every run. Defaults to ``42``.
        raw_dir (pathlib.Path | str | None): Directory for the raw CSV.
            Defaults to ``data/raw`` inside the project.
        processed_dir (pathlib.Path | str | None): Directory for the
            split CSV files. Defaults to ``data/processed``.

    Attributes:
        test_size (float): Fraction of the data held out for testing.
        random_state (int): Seed used for the train/test split.
        raw_dir (pathlib.Path): Directory holding the raw CSV file.
        processed_dir (pathlib.Path): Directory holding the split files.
        feature_names (list[str]): The 4 measurement column names.
        target_names (list[str]): The 3 species names, ordered by label.
        frame (pandas.DataFrame): The full dataset including labels, or
            ``None`` before :meth:`load` has run.
        X (numpy.ndarray): Feature matrix of shape ``(150, 4)``.
        y (numpy.ndarray): Label vector of shape ``(150,)``.
        X_train (numpy.ndarray): Training features, or ``None`` before
            :meth:`split` has run.
        X_test (numpy.ndarray): Test features, or ``None`` before
            :meth:`split` has run.
        y_train (numpy.ndarray): Training labels, or ``None`` before
            :meth:`split` has run.
        y_test (numpy.ndarray): Test labels, or ``None`` before
            :meth:`split` has run.

    Example:
        Build the default split used by this practice::

            dataset = Dataset()
            dataset.prepare()
    """

    def __init__(self, test_size=0.2, random_state=42,
                 raw_dir=None, processed_dir=None):
        self.test_size = test_size
        self.random_state = random_state
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
        self.processed_dir = (
            Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
        )

        self.feature_names = []
        self.target_names = []
        self.frame = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self, save_raw=True):
        """Load the Iris dataset into memory and optionally archive it.

        Builds a :class:`pandas.DataFrame` holding the 4 measurements,
        the integer label and a human readable species name, then
        populates :attr:`X`, :attr:`y`, :attr:`feature_names` and
        :attr:`target_names`.

        Args:
            save_raw (bool): Whether to also write the data to
                ``raw_dir/iris.csv``. Defaults to ``True``.

        Returns:
            Dataset: This same instance, so calls can be chained.

        Example:
            Load without touching the filesystem::

                dataset = Dataset().load(save_raw=False)
        """
        iris = load_iris()

        self.feature_names = list(iris.feature_names)
        self.target_names = list(iris.target_names)
        self.X = iris.data
        self.y = iris.target

        frame = pd.DataFrame(iris.data, columns=self.feature_names)
        frame[LABEL_COLUMN] = iris.target
        frame["species_name"] = frame[LABEL_COLUMN].map(
            dict(enumerate(self.target_names))
        )
        self.frame = frame

        if save_raw:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(self.raw_dir / "iris.csv", index=False)

        return self

    def split(self, save=True):
        """Split the loaded data into a train and a test set.

        The split is stratified, so each species keeps the same
        proportion in both sets, and seeded with :attr:`random_state`
        so the result is identical on every run.

        Args:
            save (bool): Whether to write ``train.csv`` and ``test.csv``
                into :attr:`processed_dir`. Defaults to ``True``.

        Returns:
            Dataset: This same instance, so calls can be chained.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.

        Example:
            Split an already loaded dataset::

                dataset = Dataset().load().split()
        """
        if self.X is None:
            raise RuntimeError("Call load() before split().")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        if save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self._save_split("train", self.X_train, self.y_train)
            self._save_split("test", self.X_test, self.y_test)

        return self

    def prepare(self, save=True):
        """Run the full data pipeline in one call.

        Convenience wrapper that calls :meth:`load` followed by
        :meth:`split`. This is what a training script normally uses.

        Args:
            save (bool): Whether to write the raw and split CSV files
                to disk. Defaults to ``True``.

        Returns:
            Dataset: This same instance, ready to be trained on.

        Example:
            One line to get usable arrays::

                dataset = Dataset().prepare()
        """
        self.load(save_raw=save)
        self.split(save=save)
        return self

    def summary(self):
        """Describe the current state of the dataset as a dictionary.

        Useful for logging, and for storing alongside a trained model so
        that an experiment can be reproduced later.

        Returns:
            dict: Keys ``n_samples``, ``n_features``, ``train_samples``,
            ``test_samples``, ``test_size``, ``random_state``,
            ``feature_names`` and ``target_names``.

        Example:
            Record how an experiment was configured::

                dataset.summary()["train_samples"]
        """
        return {
            "n_samples": 0 if self.X is None else len(self.X),
            "n_features": len(self.feature_names),
            "train_samples": 0 if self.X_train is None else len(self.X_train),
            "test_samples": 0 if self.X_test is None else len(self.X_test),
            "test_size": self.test_size,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
        }

    def _save_split(self, name, X, y):
        """Write one split to :attr:`processed_dir` as a CSV file.

        Args:
            name (str): Split name, used as the file stem, for example
                ``"train"`` or ``"test"``.
            X (numpy.ndarray): Feature matrix for this split.
            y (numpy.ndarray): Labels for this split.
        """
        frame = pd.DataFrame(X, columns=self.feature_names)
        frame[LABEL_COLUMN] = y
        frame.to_csv(self.processed_dir / f"{name}.csv", index=False)

    @classmethod
    def load_split(cls, name="test", processed_dir=None):
        """Read a previously saved split back from disk.

        Lets a notebook or a separate analysis script work with the
        exact same test set the model was evaluated on, without having
        to re-run the split.

        Args:
            name (str): Which split to read, ``"train"`` or ``"test"``.
                Defaults to ``"test"``.
            processed_dir (pathlib.Path | str | None): Directory to read
                from. Defaults to ``data/processed``.

        Returns:
            tuple: ``(X, y, feature_names)`` where *X* is a
            :class:`numpy.ndarray` of features, *y* a
            :class:`numpy.ndarray` of labels, and *feature_names* the
            list of column names.

        Raises:
            FileNotFoundError: If the requested CSV file is missing.

        Example:
            Load the held out test set::

                X_test, y_test, feature_names = Dataset.load_split("test")
        """
        directory = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
        path = directory / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'uv run python -m src.train' first."
            )

        frame = pd.read_csv(path)
        feature_names = [c for c in frame.columns if c != LABEL_COLUMN]

        X = frame[feature_names].to_numpy()
        y = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)

        return X, y, feature_names
