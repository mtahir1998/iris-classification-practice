"""Training experiment for the Iris flower classifier.

This script is one *experiment*: it takes the data from
:class:`src.dataset.Dataset`, the estimator from :class:`src.model.Model`,
performs the training, and writes the fitted model plus a metadata file
to ``models/<experiment name>/``.

The script deliberately contains no data loading and no model internals.
To run a variation, copy this file, change the constants at the top, and
run the copy. Fixes made in ``dataset.py`` or ``model.py`` are then
picked up by every experiment without editing any of them.

Example:
    Run the experiment from the project root::

        uv run python -m src.train
"""

import json
from datetime import datetime, timezone

from src.dataset import PROJECT_ROOT, Dataset
from src.model import Model

EXPERIMENT_NAME = "logistic_regression"
"""str: Name of this experiment, used as the output folder name."""

TEST_SIZE = 0.2
"""float: Fraction of the data held out for testing."""

RANDOM_STATE = 42
"""int: Seed shared by the split and the estimator for reproducibility."""

MAX_ITER = 200
"""int: Maximum number of optimisation iterations for the estimator."""


def run(experiment_name=EXPERIMENT_NAME, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, max_iter=MAX_ITER):
    """Prepare the data, fit the model, and save both model and metadata.

    The output directory ``models/<experiment_name>/`` is created if it
    does not exist and will contain ``model.joblib`` and
    ``metadata.json``.

    Args:
        experiment_name (str): Folder name for the results of this run.
            Defaults to :data:`EXPERIMENT_NAME`.
        test_size (float): Fraction of the data held out for testing.
            Defaults to :data:`TEST_SIZE`.
        random_state (int): Seed for the split and the estimator.
            Defaults to :data:`RANDOM_STATE`.
        max_iter (int): Maximum optimisation iterations for the
            estimator. Defaults to :data:`MAX_ITER`.

    Returns:
        dict: The metadata that was written to disk, including the
        train and test accuracy of the fitted model.

    Example:
        Run a variation without editing the file::

            run(experiment_name="logreg_500", max_iter=500)
    """
    dataset = Dataset(test_size=test_size, random_state=random_state).prepare()
    print(
        f"Data ready: {dataset.summary()['train_samples']} train / "
        f"{dataset.summary()['test_samples']} test samples"
    )

    model = Model(max_iter=max_iter, random_state=random_state)
    model.fit(dataset.X_train, dataset.y_train)
    print(f"Trained {model.name}")

    train_accuracy = model.score(dataset.X_train, dataset.y_train)
    test_accuracy = model.score(dataset.X_test, dataset.y_test)
    print(f"  Train accuracy: {train_accuracy:.2%}")
    print(f"  Test accuracy:  {test_accuracy:.2%}")

    output_dir = PROJECT_ROOT / "models" / experiment_name
    model_path = model.save(output_dir / "model.joblib")

    metadata = {
        "experiment": experiment_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        **model.config(),
        **dataset.summary(),
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model:    {model_path}")
    print(f"Saved metadata: {metadata_path}")

    return metadata


def main():
    """Entry point used when the module is run from the command line.

    Calls :func:`run` with the constants defined at the top of this
    module and points the reader at the analysis notebook.
    """
    run()
    print("\nNext: open notebooks/analysis.ipynb to generate the figures.")


if __name__ == "__main__":
    main()
