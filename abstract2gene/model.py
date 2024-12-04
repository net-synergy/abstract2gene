"""Models for comparing abstract features.

Various models for comparing abstract embeddings. This generally means
comparing an individual publication's abstract embeddings to template
embeddings for different labels (in the case of this package, genes).

Templates are the average of many examples of abstracts tagged with a label.
"""

__ALL__ = ["ModelNoWeights", "ModelTraditionalLoss", "ModelMaximizeDistance"]

import os
from math import e
from typing import Any, Iterator

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import LeaveOneOut

RESULT_TEMPLATE = "results/{name}_validation.tsv"


class Model:
    """Base class for abstract2gene prediction models.

    Models can be trained in order to predict how similar an abstract's LLM
    embedding is to a template.

    Learns label independent weights. The weights are used to create new
    features that are a linear combination of the original features. Since the
    are not trained to the specific labels, they can be used to improve
    prediction on labels that haven't been seen during training.

    Train the model by calling the `train` method.

    Use the model by calling the `predict` method (after training).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray[Any, np.dtype[np.bool_]],
        symbols: np.ndarray,
        seed: int,
        train_test_val: tuple[float, float, float] = (0.8, 0.1, 0.1),
        name: str = "",
        batch_size: int = 64,
        n_validation_labels: int = 10,
        n_template_samples: int = 32,
    ):
        """Initialize a model.

        Parameters
        ----------
        features, labels : numpy.ndarray
            The data to train on. Both should have the same number of rows
            (independent samples).
        symbols : numpy.ndarray
            An object type array of label names.
        seed : int
            Seed for the model's random number generator. Run `reset_rng` to
            reseed the model with the value used to create the model.
        train_test_val : tuple[float, float, float], default (0.8, 0.1, 0.1)
            Proportional of labels to use for training, testing, and
            validation.
        name : str, optional
            A name used create a results filename.
        batch_size : int, default 64
            How many examples to train on for each label. Half the examples
            will be labeled with the current label and the other half will be
            randomly selected for the pool of samples not associated with the
            current label.
        n_validation_labels : int, default 10
            Number of labels to be used for validation instead of training.
        n_template_samples : int, default 32
            How many examples to use when creating the templates for a given
            label. A label's template is created by averaging positive examples
            of features associated with the given label.

        Note: There must be at least `(batch_size // 2) + n_template_samples`
        samples tagged with each label. So labels should be filtered beforehand
        to only those with enough examples.

        """
        assert batch_size % 2 == 0, "Batch size must be even."

        self.name = name
        self.result_file = RESULT_TEMPLATE.format(name=name)
        self.seed = seed
        self.rng: np.random.Generator = default_rng(seed)
        self.masks = self._split_labels(labels, train_test_val)
        self.labels = labels
        self.features = features
        self.symbols = symbols
        self.batch_size = batch_size
        self.n_template_samples = n_template_samples
        self.weights: np.ndarray[Any, np.dtype[np.floating]] = np.asarray([0])

    def _split_labels(
        self,
        labels: np.ndarray,
        proportions: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        """Randomly sample labels to use in training, testing, and validation.

        Returns the training and test masks. Validation labels are those not in
        either training or testing.
        """
        train_size = int(proportions[0] * labels.shape[1])
        test_size = int(proportions[1] * labels.shape[1])
        val_size = labels.shape[1] - (train_size + test_size)
        mask = np.concat(
            tuple(
                np.zeros((sz)) + i
                for i, sz in enumerate((train_size, test_size, val_size))
            ),
        )
        self.rng.shuffle(mask)

        return {"train": mask == 0, "test": mask == 1, "validate": mask == 2}

    def reset_rng(self, seed: int | None = None):
        """Reset the models RNG.

        To allow reproducibly training the model, reset the random number
        generator to the original seed the model was created with.

        If `seed` provided uses this seed instead.
        """
        seed = seed or self.seed
        self.rng = default_rng(seed)

    def predict(self, x: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Calculate similarity of samples `x` to templates `template`.

        Parameters
        ----------
        x : numpy.ndarray
            Either a row vector of a single sample or a matrix where each row
            is a sample.
        template : numpy.ndarray
            Either a row vector of a single template or a matrix where each row
            is a separate template.

        Returns
        -------
        similarity : numpy.ndarray
            A n_samples x n_templates matrix with a similarity score between 0
            and 1 for each sample, template pair.

        """
        return (x @ self.weights) @ (self.weights.T @ template.T)

    def loss(self, batch) -> float:
        """Score the model's prediction success against known labels."""
        return (
            0.5
            * (
                np.square(
                    batch["labels"]
                    - self.predict(batch["samples"], batch["templates"])
                )
            ).mean()
        )

    def gradient(self, batch, weights) -> np.ndarray:
        """Calculate the loss function's gradient with respect to weights."""
        raise NotImplementedError

    def update(self, batch, learning_rate) -> None:
        """Update the model's weights based on the loss gradient."""
        wt_drop = self.weights.copy()
        wt_drop[self.dropout] = 0
        delta = learning_rate * self.gradient(batch, wt_drop)
        delta[self.dropout] = 0

        self.weights -= learning_rate * delta
        self.weights /= np.linalg.norm(self.weights, axis=0, keepdims=True)

    def batches(self, task: str = "train") -> Iterator[dict[str, np.ndarray]]:
        """Generate batches of features to train on."""
        labels = self.labels[:, self.masks[task]]
        label_pool = self.rng.permutation(np.arange(labels.shape[1]))
        for label_idx in label_pool:
            yield self._split_data(labels[:, label_idx])

    def validate(self):
        """Calculate the average validation set loss."""
        loss = 0
        count = 0
        for batch in self.batches("validate"):
            loss += self.loss(batch)
            count += 1

        return loss / count

    def _split_data(
        self,
        indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        samples = np.arange(indices.shape[0])
        samples_true = self.rng.permutation(samples[indices])
        samples_false = self.rng.permutation(samples[np.logical_not(indices)])
        mini_batch_size = self.batch_size // 2

        return {
            "samples": self.features[
                np.concat(
                    (
                        samples_true[:mini_batch_size],
                        samples_false[:mini_batch_size],
                    )
                ),
                :,
            ],
            "templates": self.features[
                samples_true[
                    mini_batch_size : (
                        mini_batch_size + self.n_template_samples
                    )
                ],
                :,
            ].mean(axis=0, keepdims=True),
            "labels": np.concat(
                (
                    np.ones((mini_batch_size, 1), dtype=np.bool_),
                    np.zeros((mini_batch_size, 1), dtype=np.bool_),
                )
            ),
        }

    def train(
        self,
        ndims: int = 10,
        max_iter: int = 1000,
        stop_delta: float = 1e-5,
        learning_rate: float = 0.002,
        learning_decay: float = 0.1,
        dropout: float = 0.2,
        verbose: bool = True,
    ):
        """Train weights for the model."""
        self.weights = self.rng.normal(
            0, 0.01, (self.features.shape[1], ndims)
        )
        self.weights /= np.linalg.norm(self.weights, axis=0, keepdims=True)
        self.dropout = self.rng.uniform(size=(self.weights.shape)) < dropout

        epoch = 0
        last_err = self.validate()
        if verbose:
            print(f"Initial validation loss: {last_err:0.4g}")

        delta = stop_delta + 1
        while (epoch < max_iter) and (delta > stop_delta) or epoch < 30:
            train_err = 0.0
            count = 0
            for batch in self.batches():
                if count == 0:
                    x_t = batch["samples"][batch["labels"].squeeze(), :]
                    x_f = batch["samples"][
                        np.logical_not(batch["labels"].squeeze()), :
                    ]
                    print(
                        f"  True: {self.predict(x_t, batch["templates"]).mean()}"
                    )
                    print(
                        f"  False: {self.predict(x_f, batch["templates"]).mean()}"
                    )
                self.update(batch, learning_rate)
                train_err += self.loss(batch)
                count += 1

            self.dropout = (
                self.rng.uniform(size=(self.weights.shape)) < dropout
            )
            learning_rate *= e ** (-learning_decay)
            err = self.validate()
            if verbose:
                print(f"Epoch: {epoch}")
                print(f"  Training loss: {train_err / count:.4g}")
                print(f"  Validation loss: {err:.4g}")
                print(f"  Delta: {(last_err - err):.4g}")

            delta = abs(last_err - err)
            last_err = err
            epoch += 1

    def test(self, max_num_tests: int | None = None):
        """Tests how well the model differentiates labels.

        For each label, compares predictions for `self.batch_size` labeled
        samples against samples without the label. Templates are created using
        a leave-one-out method. Results are stored as a table in
        `self.result_file`.

        The `max_num_tests` controls how many labels to test. If None, tests
        are test labels.

        """
        if os.path.exists(self.result_file):
            os.unlink(self.result_file)

        labels_test = self.labels[:, self.masks["test"]]
        symbols_test = self.symbols[self.masks["test"]]

        max_num_tests = max_num_tests or labels_test.shape[1]
        n_tests = min(max_num_tests, labels_test.shape[1])

        loss = 0.0
        for i in range(n_tests):
            loss += self._test_label(labels_test[:, i], symbols_test[i])

        print(f"Average sample mean distance:\n  {loss / n_tests:0.4g}")

    def _test_label(self, indices, symbol):
        features_label = self.features[indices, :]
        # Make all tests have same number of samples.
        features_label = features_label[: self.batch_size, :]
        features_other = self.features[np.logical_not(indices), :]

        loo = LeaveOneOut()
        sim_within = np.zeros((self.batch_size))
        sim_between = np.zeros((self.batch_size))
        for i, (train_index, test_index) in enumerate(
            loo.split(features_label)
        ):
            template = features_label[train_index, :].mean(
                axis=0, keepdims=True
            )
            sim_within[i] = self.predict(
                features_label[test_index, :], template
            )[0, 0]
            sim_between[i] = self.predict(features_other[[i], :], template)[
                0, 0
            ]

        label_df = pd.DataFrame(
            {
                "label": np.repeat(symbol, self.batch_size * 2),
                "group": np.concat(
                    (
                        np.asarray("within").repeat(self.batch_size),
                        np.asarray("between").repeat(self.batch_size),
                    )
                ),
                "similarity": np.concat((sim_within, sim_between)),
            }
        )

        header = not os.path.exists(self.result_file)
        label_df.to_csv(
            self.result_file, sep="\t", index=False, mode="a", header=header
        )

        distance = sim_within.mean() - sim_between.mean()
        stderr = np.concat((sim_within, sim_between)).std(ddof=1)
        stderr /= self.batch_size * 2
        return distance / stderr


class ModelNoWeights(Model):
    """Model without weights.

    Prediction is the dot product between the samples and templates.
    """

    def train(self, *args, **kwds):
        pass

    def predict(self, x, template) -> np.ndarray:
        return x @ template.T


class ModelTraditionalLoss(Model):
    """Model based on the mean squared error of y_hat - y loss function."""

    def gradient(self, batch, weights) -> np.ndarray:
        """Calculate gradient of loss function with respect to weights.

        Note: labels are assumed to be a vector. When using the model itself
        labels will likely be a matrix of samples x n_genes. When training only
        one gene is tested at a given time. Due to the math, `x`, `template`,
        and `labels` must all be vectors while `weights` is a matrix. It is
        expected that `x` is actually a samples x features matrix and the
        matrix operations are iterated over it's samples.
        """
        return (
            sum(
                (self.predict(x.reshape((1, -1)), batch["templates"]) - label)
                * (
                    x.reshape((-1, 1)) @ (batch["templates"] @ weights)
                    + batch["templates"].T @ (x.reshape((1, -1)) @ weights)
                )
                for x, label in zip(batch["samples"], batch["labels"])
            )
            / batch["samples"].shape[0]
        )


class ModelMaximizeDistance(Model):
    """Maximizes the distance between labeled and unlabeled predictions.

    Instead of comparing directly to the boolean label, compare sample mean of
    data labeled with a given label to the null (data not labeled with the
    current label) sample mean.
    """

    def predict(self, x, template):
        ndims = self.weights.shape[1]
        return super().predict(x, template) / ndims

    def loss(self, batch):
        def per_label_loss(x, template, label):
            x_true = x[label, :]
            x_false = x[np.logical_not(label), :]
            y_hat_true = self.predict(x_true, template)
            y_hat_false = self.predict(x_false, template)

            return np.square(1 - (y_hat_true - y_hat_false)).mean()

        return (
            0.5
            * sum(
                per_label_loss(batch["samples"], t.reshape(1, -1), l)
                for t, l in zip(batch["templates"], batch["labels"].T)
            )
            / batch["labels"].shape[1]
        )

    def gradient(self, batch, weights):
        x_true = batch["samples"][batch["labels"].squeeze(), :]
        x_false = batch["samples"][
            np.logical_not(batch["labels"].squeeze()), :
        ]

        return (
            sum(
                (
                    1
                    - (
                        self.predict(x_t.reshape((1, -1)), batch["templates"])[
                            0, 0
                        ]
                        - self.predict(
                            x_f.reshape((1, -1)), batch["templates"]
                        )[0, 0]
                    )
                )
                * (
                    (
                        x_f.reshape((-1, 1)) @ (batch["templates"] @ weights)
                        + batch["templates"].T
                        @ (x_f.reshape((1, -1)) @ weights)
                    )
                    - (
                        x_t.reshape((-1, 1)) @ (batch["templates"] @ weights)
                        + batch["templates"].T
                        @ (x_t.reshape((1, -1)) @ weights)
                    )
                )
                for x_t, x_f in zip(x_true, x_false)
            )
            / x_true.shape[0]
        )
