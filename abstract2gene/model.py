__ALL__ = ["split_genes", "train"]

from math import e
from typing import Any, Callable, Iterator

import numpy as np
from numpy.random import default_rng

from .typing import ArrayLike


def split_genes(
    labels: ArrayLike, train_size: float = 0.8, seed: int | None = None
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Randomly sample genes to use in training.

    Returns a mask for indexing the labels as `labels[:, mask]` to get training
    genes and `labels[:, np.logical_not(mask)]` for test genes.
    """
    n_training_samples = int(train_size * labels.shape[1])
    training_mask = np.fromiter(
        (1 if i < n_training_samples else 0 for i in range(labels.shape[1])),
        dtype=np.bool_,
    )
    rng = default_rng(seed)
    rng.shuffle(training_mask)

    return training_mask


class model:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        n_weight_dims: int = 10,
        n_validation_labels: int = 10,
        learning_rate: float = 0.01,
        learning_decay: float = 0.0,
        seed: int = 0,
    ):
        self.rng: np.random.Generator = default_rng(seed)
        self.batch_size = batch_size
        self.validation_set, self.validation_labels = (
            self._generate_validation_set(
                features, labels[:, -n_validation_labels:]
            )
        )
        self.features = features
        self.n_features = features.shape[1]
        self.labels = labels[:, :-n_validation_labels]
        self.n_labels = self.labels.shape[1]
        self.lr = learning_rate
        self.decay = learning_decay
        self.weights = self.rng.normal(
            0, 0.01, (self.n_features, n_weight_dims)
        )
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)
        self._acc_loss = 0.0

    def _generate_validation_set(
        self, features, labels
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        val_idx = np.any(labels, axis=1)
        val_features = features[val_idx, :]
        val_labels = labels[val_idx, :]
        count = np.cumsum(val_labels, axis=0)
        template_idx = np.logical_and(count < self.batch_size, val_labels)
        templates = np.fromiter(
            (
                val_features[template_idx[:, i], :].mean(axis=0)
                for i in range(val_labels.shape[1])
            ),
            dtype=np.dtype((val_features.dtype, val_features.shape[1])),
        )
        val_features = val_features[
            np.logical_not(np.any(template_idx, axis=1))
        ]
        val_labels = val_labels[np.logical_not(np.any(template_idx, axis=1))]

        return ((val_features, templates), val_labels)

    def predict(self, x, template) -> np.ndarray:
        return (x @ self.weights) @ (self.weights.T @ template.T)

    def loss(self, x, template, labels) -> np.ndarray:
        return 0.5 * (np.square(labels - self.predict(x, template))).mean()

    def gradient(self, x, template, labels) -> np.ndarray:
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
                (self.predict(sample.reshape((1, -1)), template) - label)
                * (
                    sample.reshape((-1, 1)) @ (template @ self.weights)
                    + template.T @ (sample.reshape((1, -1)) @ self.weights)
                )
                for sample, label in zip(x, labels)
            )
            / x.shape[0]
        )

    def update(self, x, template, labels):
        self.weights -= self.lr * self.gradient(x, template, labels)
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)
        self._acc_loss += self.loss(x, template, labels)

    def print_status(self):
        print(
            f"training_err: {self._acc_loss / self.n_labels}\t"
            + "validation_err: "
            + f"{self.loss(*self.validation_set, self.validation_labels)}"
        )

    def batches(
        self,
    ) -> Iterator[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
        step = 0
        label_pool = np.arange(self.n_labels)
        self.rng.shuffle(label_pool)
        self._acc_loss = 0.0
        while step < label_pool.shape[0]:
            label_idx = label_pool[step]
            yield self._split_data(self.labels[:, label_idx])
            step += 1
        self.lr *= e ** (-self.decay)

    def _split_data(
        self,
        indices: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        samples = np.arange(indices.shape[0])
        samples_true = self.rng.permutation(samples[indices])
        samples_false = self.rng.permutation(samples[np.logical_not(indices)])
        mini_batch_size = self.batch_size // 8

        return (
            (
                # Sample of features. Half with current label, half without.
                self.features[
                    np.concat(
                        (
                            samples_true[:mini_batch_size],
                            samples_false[: (7 * mini_batch_size)],
                        )
                    ),
                    :,
                ],
                # Template for current label.
                self.features[
                    samples_true[mini_batch_size : (8 * mini_batch_size)], :
                ].mean(axis=0, keepdims=True),
            ),
            # Labels for batch's samples.
            np.concat(
                (
                    np.ones((mini_batch_size, 1), dtype=np.bool_),
                    np.zeros((7 * mini_batch_size, 1), dtype=np.bool_),
                )
            ),
        )


def train(
    features: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
    ndims: int = 10,
    batch_size: int = 64,
    max_iter: int = 1000,
    stop_delta: float = 1e-5,
    learning_rate: float = 0.002,
    verbose: bool = True,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Calculate weights for the model."""
    assert batch_size % 2 == 0, "Batch size must be even."

    ml = model(
        features,
        labels,
        batch_size,
        ndims,
        10,
        learning_rate,
        0.1,
        random_seed,
    )

    epoch = 0
    last_err = ml.loss(*ml.validation_set, ml.validation_labels)
    if verbose:
        print(f"Epoch: {-1}, delta: {last_err}")
    delta = stop_delta + 1
    while (epoch < max_iter) and (delta > stop_delta):
        for (samples_train, template), sample_labels in ml.batches():
            ml.update(samples_train, template, sample_labels)

        if verbose:
            ml.print_status()

        err = ml.loss(*ml.validation_set, ml.validation_labels)
        delta = abs(last_err - err)
        last_err = err
        if verbose:
            print(f"Epoch: {epoch}, delta: {delta}")
        epoch += 1

    return ml.predict
