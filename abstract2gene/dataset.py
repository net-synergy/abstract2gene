from __future__ import annotations

from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np
from pubnet import PubNet

from .typing import ArrayLike, Batch, LabelLike

__ALL__ = ["net2dataset"]


def net2dataset(
    net: PubNet,
    features: str = "Abstract_embedding",
    labels: str = "Gene",
    label_name: str = "GeneSymbol",
    feature_name: str = "PMID",
    min_occurrences: int = 50,
    remove_baseline: bool = False,
    arrays: str = "jax",
    batch_size: int = 64,
    template_size: int = 32,
    seed: int = 0,
    train_test_val_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
    axis: int = 1,
) -> DataSet:
    """Return the network as a matrix of features and a matrix of labels.

    Features are normalized by publication so each publication's features has
    an l2 norm of 1.

    Features are samples (publications) x features and labels are samples x
    labels (binary vector).

    Label IDs is an array of the label nodes index that are in used in the
    labels. Labels are limited to genes with at least `min_occurrences`.

    If `remove_baseline` subtract the average of each feature.
    """
    feature_names = net.get_node("Publication").feature_vector(feature_name)
    embeddings_edge = net.get_edge(features, "Publication")
    n_features = np.sum(embeddings_edge["Publication"] == 0)
    embeddings = embeddings_edge.feature_vector("embedding").reshape(
        (-1, n_features)
    )

    if remove_baseline:
        baseline = embeddings.mean(axis=0, keepdims=True)
        embeddings = embeddings - baseline

    embeddings = embeddings / np.reshape(
        np.linalg.norm(embeddings, axis=1), shape=(-1, 1)
    )

    label_edges = net.get_edge("Publication", labels)
    label_frequencies = np.unique_counts(label_edges[labels])
    locs = label_frequencies.counts >= min_occurrences
    frequent_labels = label_frequencies.values[locs]
    label_edges = label_edges[label_edges.isin(labels, frequent_labels)]
    label_nodes = net.get_node(labels).loc(frequent_labels)
    label_map = dict(
        zip(label_nodes.index, np.arange(frequent_labels.shape[0]))
    )

    label_vec: LabelLike = np.zeros(
        (embeddings.shape[0], frequent_labels.shape[0]), np.bool_
    )
    label_vec[
        label_edges["Publication"],
        np.fromiter((label_map[l] for l in label_edges[labels]), dtype=int),
    ] = True

    label_names = label_nodes.feature_vector(label_name)

    if arrays == "jax":
        embeddings = jnp.asarray(embeddings)
        label_vec = jnp.asarray(label_vec)

    return DataSet(
        embeddings,
        label_vec,
        feature_names,
        label_names,
        train_test_val_split,
        batch_size,
        template_size,
        seed,
        axis,
    )


class DataSet:
    def __init__(
        self,
        features: ArrayLike,
        labels: LabelLike,
        feature_names: np.ndarray | None,
        label_names: np.ndarray | None,
        train_test_val_split: tuple[float, float, float],
        batch_size: int,
        template_size: int,
        seed: int,
        axis: int,
    ):
        self.features = features
        self.labels = labels
        self.feature_names = feature_names
        self.label_names = label_names
        self.batch_size = batch_size
        self.template_size = template_size
        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._axis = axis
        self.train_test_val_split = train_test_val_split
        self._index = 0

        self.reshuffle()

    def reshuffle(self) -> None:
        """Resample the data to get a new train-test-validate split.

        Reshuffles the masks as well as sample and label order.

        Since the features and labels get shuffled, resetting the RNG before
        shuffling will not reproduce the original state. The only way to
        reproduce the original state after the data has been shuffled is to
        recreate the dataset with the same seed and original data.
        """
        mix_feats_idx = self._rng.permutation(self.n_samples)
        self.features = self.features[mix_feats_idx, :]
        self.labels = self.labels[mix_feats_idx, :]

        mix_label_idx = self._rng.permutation(self.n_labels)
        self.labels = self.labels[:, mix_label_idx]

        if self.feature_names is not None:
            self.feature_names = self.feature_names[mix_feats_idx]

        if self.label_names is not None:
            self.label_names = self.label_names[mix_label_idx]

        self._masks = self._make_masks(self._split)

    def _make_masks(
        self, split: tuple[float, float, float]
    ) -> dict[str, LabelLike]:
        """Randomly sample labels to use in training, testing, and validation.

        Returns the training and test masks. Validation labels are those not in
        either training or testing.
        """
        n = self.n_samples if self._axis == 0 else self.n_labels

        train_size = int(n * split[0])
        test_size = int(n * split[1])
        val_size = n - (train_size + test_size)
        mask = np.concat(
            tuple(
                np.zeros((sz)) + i
                for i, sz in enumerate((train_size, test_size, val_size))
            ),
        )

        return {"train": mask == 0, "test": mask == 1, "validate": mask == 2}

    def reset_rng(self, seed: int | None = None):
        """Reset the RNG to the original seed.

        If `seed` is provided use instead of the original seed.
        """
        seed = seed or self._seed
        self._rng = np.random.default_rng(seed)

    @property
    def train_test_val_split(self):
        return self._split

    @train_test_val_split.setter
    def train_test_val_split(self, new_split: tuple[float, float, float]):
        """Provide new proportions to split the data then reshuffle."""
        self._split = new_split
        self.reshuffle()

    def get_symbol(self) -> str | None:
        """Return the current batch's symbol."""
        if self.label_names is None:
            return None
        return self.label_names[self._index]

    @property
    def n_features(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.features.shape[1]

    @property
    def n_labels(self) -> int:
        """Return number of unique labels."""
        return self.labels.shape[1]

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return self.features.shape[0]

    @property
    def n_train(self) -> int:
        """Return the size of the training set."""
        return self._masks["train"].sum()

    @property
    def n_test(self) -> int:
        """Return the size of the testing set."""
        return self._masks["test"].sum()

    @property
    def n_validate(self) -> int:
        """Return the size of the validation set."""
        return self._masks["validate"].sum()

    def train(self, **kwds) -> Iterable[Batch]:
        """Return random batches of training data."""
        return self._batch("train", **kwds)

    def test(self, **kwds) -> Iterable[Batch]:
        """Return random batches of testing data."""
        return self._batch("test", **kwds)

    def validate(self, **kwds) -> Iterable[Batch]:
        """Return random batches of validation data."""
        return self._batch("validate", **kwds)

    def _batch(
        self,
        task: str,
        batch_size: int | None = None,
        template_size: int | None = None,
    ) -> Iterable[Batch]:
        """Generate batches of features to train on."""
        batch_size = batch_size or self.batch_size
        template_size = template_size or self.template_size

        def _batch_rows(task, batch_size, template_size):
            n_labels = 100
            samples = self._rng.permutation(
                np.arange(self.n_samples)[self._masks[task]]
            )
            labels = self.labels[samples, :]
            occurances = labels.cumsum(axis=0)
            labels = labels[
                :, occurances[-1, :] >= (self.template_size + self.batch_size)
            ]
            occurances = occurances[
                :, occurances[-1, :] >= (self.template_size + batch_size)
            ]
            n_labels = min(n_labels, labels.shape[1])
            label_idx = self._rng.choice(
                labels.shape[1], n_labels, replace=False
            )
            templates = np.fromiter(
                (
                    self.features[
                        samples[occurances[:, i] < self.template_size],
                        :,
                    ].mean(axis=0, keepdims=False)
                    for i in label_idx
                ),
                dtype=(self.features.dtype, self.n_features),
            )
            labels = labels[:, label_idx]

            samples = samples[np.all(occurances >= self.template_size, axis=1)]
            n_batches = samples.shape[0] // batch_size

            if n_batches < 1:
                raise RuntimeError("Not enough samples to create a batch.")

            for self._index in range(n_batches):
                start_idx = self._index * batch_size
                batch_samples = samples[start_idx : (start_idx + batch_size)]
                yield (
                    self.features[batch_samples, :],
                    templates,
                    labels[batch_samples, :],
                )

        def _batch_columns(task, batch_size, template_size):
            labels = self.labels[:, self._masks[task]]
            label_pool = self._rng.permutation(np.arange(labels.shape[1]))
            for label_idx in label_pool:
                self._index = label_idx
                yield self._split_data(
                    labels[:, label_idx], batch_size, template_size
                )

        if self._axis == 0:
            return _batch_rows(task, batch_size, template_size)

        return _batch_columns(task, batch_size, template_size)

    def _split_data(
        self,
        indices: LabelLike,
        batch_size: int,
        template_size: int,
    ) -> Batch:
        samples = np.arange(indices.shape[0])
        samples_true = self._rng.permutation(samples[indices])
        samples_false = self._rng.permutation(samples[np.logical_not(indices)])
        mini_batch_size = batch_size // 2

        return (
            self.features[
                np.concat(
                    (
                        samples_true[:mini_batch_size],
                        samples_false[:mini_batch_size],
                    )
                ),
                :,
            ],
            self.features[
                samples_true[
                    mini_batch_size : (mini_batch_size + template_size)
                ],
                :,
            ].mean(axis=0, keepdims=True),
            indices[
                np.concat(
                    (
                        samples_true[:mini_batch_size],
                        samples_false[:mini_batch_size],
                    )
                ).reshape((mini_batch_size * 2, -1))
            ],
        )
