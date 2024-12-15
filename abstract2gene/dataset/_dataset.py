"""Collect feature data and provide iterable sets."""

from __future__ import annotations

__all__ = ["DataSet", "load_dataset", "list_datasets", "delete_dataset"]

import os
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

import abstract2gene.storage as storage
from abstract2gene.storage import _storage_factory, default_data_dir

from ..typing import Batch, Features, Labels, Names

InFeatures = np.ndarray[Any, np.dtype[np.double]]
InLabels = np.ndarray[Any, np.dtype[np.bool_]]

_NAME = "dataset"
_DEFALUT_DATA_DIR = default_data_dir(_NAME)

list_datasets = _storage_factory(storage.list_cache, _NAME)
delete_dataset = _storage_factory(storage.delete_from_cache, _NAME)


class DataSet:
    """Collects features and labels.

    Data is split into training, testing, and validation sets and batches
    (features, labels) can be iterated over with the corresponding methods:
    `test`, `train`, `validate`.

    Labels for each batch is a column vector of batch_size x 1.
    Label is True for half the samples and False for the other half.

    Example:
    -------
    data = a2g.datasets.bioc2dataset(range(10, 20))
    for batch in data.train():
       train_step(model, batch)

    See Also:
    --------
    Suggested methods for creating a dataset.
    `abstract2gene.datasets.net2dataset`
    `abstract2gene.datasets.bioc2dataset`

    For storing templates.
    `abstract2gene.datasets.Template`

    """

    def __init__(
        self,
        features: InFeatures,
        labels: InLabels,
        feature_names: Names,
        label_names: Names,
        train_test_val_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        batch_size: int = 64,
        template_size: int = 32,
        seed: int = 0,
    ):
        """Construct a DataSet.

        Parameters
        ----------
        features : ndarray[Any, dtype[double]]
            Should be in the form n_samples x n_features
        labels : ndarray[Any, dtype[bool]]
            Should be in the form n_samples x n_labels
        feature_names : ndarray (dtype str_ or object)
            One dimensional array of feature names (such as PMIDs)
        label_names : ndarray (dtype str_ or object)
            One dimensional array of label names (such as gene symbols)
        train_test_val_split : tuple[float, float, float], Optional
            The proportion of all samples to be used for training, testing, and
            validation. Should add up to 1. Labels are split into sets.
        batch_size : int, default 64
            How many samples should make up a batch.
        template_size : int
            The number of samples averaged together to make a template.
        seed : int, default 0
            The seed for the random number generator.

        """
        self.features = jnp.asarray(features)
        self.feature_names = feature_names
        self.labels = jnp.asarray(labels)
        self.label_names = label_names

        # Set hidden properties to prevent repeatedly triggering reshuffle
        # before dataset fully initialized.
        self._batch_size = batch_size
        self._template_size = template_size
        self._split = train_test_val_split

        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._feature_names = np.asarray([""], dtype=np.dtype(np.str_))
        self._label_name = np.asarray([""], dtype=np.dtype(np.str_))[0]

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

        self.feature_names = self.feature_names[mix_feats_idx]
        self.label_names = self.label_names[mix_label_idx]

        self._masks = self._make_masks(self._split)

    def _make_masks(
        self, split: tuple[float, float, float]
    ) -> dict[str, np.ndarray[Any, np.dtype[np.bool_]]]:
        """Randomly sample labels to use in training, testing, and validation.

        Returns the training and test masks. Validation labels are those not in
        either training or testing.
        """
        tol = 1e-5
        assert sum(split) > (1 - tol) and sum(split) < (1 + tol)

        min_occurance = self.batch_size + self.template_size
        label_mask = self.labels.sum(axis=0) >= min_occurance

        train_size = int(self.n_labels * split[0])
        test_size = int(self.n_labels * split[1])
        val_size = self.n_labels - (train_size + test_size)

        mask = np.zeros((self.labels.shape[1]))
        mask[label_mask] = np.concat(
            tuple(
                np.ones(sz) + i
                for i, sz in enumerate((train_size, test_size, val_size))
            )
        )
        mask[label_mask] = self._rng.permutation(mask[label_mask])

        return {"train": mask == 1, "test": mask == 2, "validate": mask == 3}

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs: int):
        self._batch_size = bs
        self.reshuffle()

    @property
    def template_size(self) -> int:
        return self._template_size

    @template_size.setter
    def template_size(self, ts: int):
        self._template_size = ts
        self.reshuffle()

    def reset_rng(self, seed: int | None = None) -> None:
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

    @property
    def n_features(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.features.shape[1]

    @property
    def n_labels(self) -> int:
        """Return number of unique labels.

        Note this is the number of labels with at least batch_size +
        template_size samples not the raw number of columns in the label
        matrix. As such this value can change if batch size and template size
        are reset.
        """
        min_occurance = self.batch_size + self.template_size
        label_mask = self.labels.sum(axis=0) >= min_occurance
        return sum(label_mask)

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

    def batch_label_name(self) -> str:
        """Return the current batch's label name."""
        return self._label_name

    def batch_feature_names(self) -> Names:
        """Return the names for the batch's samples."""
        return self._feature_names

    def train(self) -> Iterable[Batch]:
        """Return random batches of training data."""
        return self._batch("train")

    def test(self) -> Iterable[Batch]:
        """Return random batches of testing data."""
        return self._batch("test")

    def validate(self) -> Iterable[Batch]:
        """Return random batches of validation data."""
        return self._batch("validate")

    def _batch(
        self,
        task: str,
    ) -> Iterable[Batch]:
        """Generate batches of features to train on."""
        labels = self.labels[:, self._masks[task]]
        label_pool = self._rng.permutation(np.arange(labels.shape[1]))
        for label_idx in label_pool:
            self._label_name = self.label_names[label_idx]
            yield self._split_labels(labels[:, label_idx], self.batch_size)

    def _split_labels(
        self,
        indices: Labels,
        batch_size: int,
    ) -> Batch:
        samples = np.arange(indices.shape[0])
        samples_true = self._rng.permutation(samples[indices])
        samples_false = self._rng.permutation(samples[np.logical_not(indices)])
        mini_batch_size = batch_size // 2

        self._feature_names = self.feature_names[
            np.concat(
                (
                    samples_true[:mini_batch_size],
                    samples_false[:mini_batch_size],
                )
            ),
        ]

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
                    mini_batch_size : (mini_batch_size + self.template_size)
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

    def get_templates(self) -> Features:
        """Return a matrix of templates created from dataset.

        Randomly pulls `self.template_size` samples for each label, averages
        them together, and returns the templates. Use `self.label_names` to get
        the corresponding names.
        """
        ts = self.template_size
        index = self._rng.permuted(
            self.labels
            * (np.arange(self.labels.shape[0]) + 1).reshape((-1, 1)),
            axis=0,
        )
        col_indices = (
            index[index[:, i] > 0, i] - 1 for i in range(index.shape[1])
        )

        dtype = np.dtype((self.features.dtype, (ts, self.features.shape[1])))
        return jnp.asarray(
            np.fromiter(
                (self.features[idx[:ts], :] for idx in col_indices),
                count=self.labels.shape[1],
                dtype=dtype,
            ).mean(axis=1)
        )

    def save(self, name: str, data_dir: str | None = None) -> None:
        """Save the dataset to disk.

        Parameters
        ----------
        name : str
            The name of the model (without or without ".npy" file extension).
        data_dir : str, optional
            Where to store the dataset if not using the default data directory.

        """
        data_dir = data_dir or _DEFALUT_DATA_DIR
        with open(os.path.join(data_dir, name), "wb") as f:
            np.save(f, self.features)
            np.save(f, self.labels)
            np.save(f, self.feature_names)
            np.save(f, self.label_names)


def load_dataset(name: str, data_dir: str | None = None, **kwds) -> DataSet:
    """Load a dataset from disk.

    Note only the actual data (features, labels, and names) are saved. Other
    keyword must be set again (such as batch_size, template_size, splits, and
    random seed).

    Parameters
    ----------
    name : str
        The name of the model (with or without ".npy" extension).
    data_dir : str, optional
        Where to save the file if not using the default data directory.
    **kwds :
        All other keywords are passed to the dataset constructor.

    Return
    ------
    The reconstructed dataset.

    """
    data_dir = data_dir or _DEFALUT_DATA_DIR
    with open(os.path.join(data_dir, name), "rb") as f:
        return DataSet(np.load(f), np.load(f), np.load(f), np.load(f), **kwds)
