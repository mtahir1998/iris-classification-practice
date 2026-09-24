"""Model definition for the Iris classification project.

This module owns everything related to the *estimator*: how it is
configured, how it is fitted, and how it is written to and read back
from disk. It is a thin wrapper around
:class:`sklearn.linear_model.LogisticRegression`, which keeps the
training script free of scikit-learn details and makes it possible to
swap the underlying algorithm in exactly one place.

Example:
    Fit a model and save it::

        from src.model import Model

        model = Model(max_iter=200)
        model.fit(X_train, y_train)
        model.save("models/logistic_regression.joblib")
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Model:
    """A Logistic Regression classifier for the Iris dataset.

    Despite its name, Logistic Regression is a classification
    algorithm: it learns a linear decision boundary between the
    classes and reports a probability for each one.

    The wrapper keeps the public surface small on purpose. Every
    experiment interacts with :meth:`fit`, :meth:`predict`,
    :meth:`predict_proba` and :meth:`score`, so replacing the
    estimator below changes all experiments at once.

    Args:
        max_iter (int): Maximum number of optimisation iterations
            allowed before fitting stops. Defaults to ``200``.
        random_state (int): Seed passed to the estimator so repeated
            runs give identical results. Defaults to ``42``.
        **kwargs: Any further keyword arguments accepted by
            :class:`sklearn.linear_model.LogisticRegression`, for
            example ``C`` or ``solver``.

    Attributes:
        name (str): Human readable name of the estimator, stored in
            the model metadata.
        max_iter (int): Maximum number of optimisation iterations.
        random_state (int): Seed used by the estimator.
        params (dict): Extra keyword arguments given to the estimator.
        estimator (sklearn.linear_model.LogisticRegression): The
            wrapped scikit-learn estimator that does the real work.
        is_fitted (bool): Whether :meth:`fit` has been called.

    Example:
        Train with a larger iteration budget::

            model = Model(max_iter=500)
            model.fit(X_train, y_train)
    """

    name = "LogisticRegression"

    def __init__(self, max_iter=200, random_state=42, **kwargs):
        self.max_iter = max_iter
        self.random_state = random_state
        self.params = kwargs
        self.is_fitted = False

        self.estimator = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X, y):
        """Fit the estimator on the training data.

        Args:
            X (numpy.ndarray): Training features of shape
                ``(n_samples, n_features)``.
            y (numpy.ndarray): Training labels of shape
                ``(n_samples,)``.

        Returns:
            Model: This same instance, so calls can be chained.

        Example:
            Fit and immediately score::

                model = Model().fit(X_train, y_train)
        """
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict a class label for each sample.

        Args:
            X (numpy.ndarray): Features of shape
                ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Predicted integer labels of shape
            ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted or loaded.

        Example:
            Predict the first test sample::

                model.predict(X_test[:1])
        """
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """Predict a probability for every class.

        Args:
            X (numpy.ndarray): Features of shape
                ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Probabilities of shape
            ``(n_samples, n_classes)``, each row summing to 1.

        Raises:
            RuntimeError: If the model has not been fitted or loaded.

        Example:
            Inspect the confidence of the first prediction::

                model.predict_proba(X_test[:1])
        """
        self._check_fitted()
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        """Compute accuracy on the given data.

        Args:
            X (numpy.ndarray): Features to evaluate on.
            y (numpy.ndarray): True labels for those features.

        Returns:
            float: Fraction of correct predictions, between ``0.0``
            and ``1.0``.

        Raises:
            RuntimeError: If the model has not been fitted or loaded.

        Example:
            Measure test accuracy::

                model.score(X_test, y_test)
        """
        self._check_fitted()
        return accuracy_score(y, self.predict(X))

    def config(self):
        """Report the hyperparameters this model was built with.

        Returns:
            dict: Keys ``model``, ``max_iter``, ``random_state`` and
            any extra estimator arguments that were supplied.

        Example:
            Store the configuration next to the results::

                model.config()["max_iter"]
        """
        return {
            "model": self.name,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            **self.params,
        }

    def save(self, path):
        """Write the fitted estimator to disk with joblib.

        Any missing parent directories are created automatically.

        Args:
            path (pathlib.Path | str): Destination file, conventionally
                ending in ``.joblib``.

        Returns:
            pathlib.Path: The path the model was written to.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Example:
            Save after training::

                model.save("models/logistic_regression.joblib")
        """
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)

        return path

    @classmethod
    def load(cls, path):
        """Rebuild a model from a file written by :meth:`save`.

        Args:
            path (pathlib.Path | str): Path to the saved ``.joblib``
                file.

        Returns:
            Model: An instance wrapping the restored estimator, ready
            to predict.

        Raises:
            FileNotFoundError: If no file exists at *path*.

        Example:
            Reload a trained model in a notebook::

                model = Model.load("models/logistic_regression.joblib")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'uv run python -m src.train' first."
            )

        estimator = joblib.load(path)

        model = cls(
            max_iter=getattr(estimator, "max_iter", 200),
            random_state=getattr(estimator, "random_state", 42),
        )
        model.estimator = estimator
        model.is_fitted = True

        return model

    def _check_fitted(self):
        """Raise if the estimator has not been fitted or loaded.

        Raises:
            RuntimeError: If :attr:`is_fitted` is ``False``.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or Model.load() before using the model.")
