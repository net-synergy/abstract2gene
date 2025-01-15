"""Collect feature data and provide iterable sets."""

from __future__ import annotations

__all__ = [
    "DataLoader",
    "from_huggingface",
    "load_dataset",
]

import os
from typing import Any, Iterable, TypeAlias

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from abstract2gene.storage import default_data_dir

from ..typing import Batch, Features, Labels, Names

InFeatures: TypeAlias = np.ndarray[Any, np.dtype[np.double]] | jax.Array
InLabels: TypeAlias = sp.sparse.csc_array


class DataLoader:
    """Collects features and labels.

    Data is split into training, testing, and validation sets and batches
    (features, labels) can be iterated over with the corresponding methods:
    `test`, `train`, `validate`.

    Batch data consists only of labeled data. To get unlabeled data, use
    `self.unlabeled`.

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
        sample_names: Names,
        label_names: Names,
        train_test_val_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        batch_size: int = 64,
        template_size: int = 32,
        seed: int = 0,
    ):
        """Construct a DataLoader.

        Parameters
        ----------
        features : ndarray[Any, dtype[double]]
            Should be in the form n_samples x n_features
        labels : 2d sparse array (csc form)
            Should be in the form n_samples x n_labels.
        sample_names : ndarray (dtype str_ or object)
            One dimensional array of feature names (such as PMIDs)
        label_names : ndarray (dtype str_ or object)
            A vector of names for the columns of labels.
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
        self.sample_names = sample_names
        self.labels = labels
        self.label_names = label_names

        # Set hidden properties to prevent repeatedly triggering reshuffle
        # before dataloader fully initialized.
        self._batch_size = batch_size
        self._template_size = template_size
        self._split = train_test_val_split

        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._sample_names = np.asarray([""], dtype=np.dtype(np.str_))
        self._label_name = np.asarray([""], dtype=np.dtype(np.str_))[0]

        self._masks: dict[str, np.ndarray[Any, np.dtype[np.bool]]] = {}
        self._unlabeled_mask: np.ndarray[Any, np.dtype[np.bool]] = np.asarray(
            []
        )
        self.reshuffle()

    def reshuffle(self) -> None:
        """Reshuffle the training, test, and validation sets."""
        tol = 1e-5
        split = self._split

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

        self._masks = {
            "train": mask == 1,
            "test": mask == 2,
            "validate": mask == 3,
        }

        self._unlabeled_mask = self.labels.sum(axis=1) == 0

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

    @property
    def unlabeled(self) -> jax.Array:
        """Return features for unlabeled samples."""
        return self.features[self._unlabeled_mask, :]

    def batch_label_name(self) -> str:
        """Return the current batch's label name."""
        return self._label_name

    def batch_sample_names(self) -> Names:
        """Return the names for the batch's samples."""
        return self._sample_names

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
            self._label_name = self.label_names[self._masks[task]][label_idx]
            yield self._split_labels(
                labels[:, [label_idx]].toarray().squeeze(), self.batch_size
            )

    def _split_labels(
        self,
        indices: Labels,
        batch_size: int,
    ) -> Batch:
        samples = np.arange(indices.shape[0])
        samples_true = self._rng.permutation(samples[indices])
        samples_false = self._rng.permutation(
            samples[
                np.logical_not(np.logical_or(indices, self._unlabeled_mask))
            ]
        )
        mini_batch_size = batch_size // 2

        self._sample_names = self.sample_names[
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
            jnp.concat(
                (
                    jnp.ones((mini_batch_size, 1), dtype=jnp.bool),
                    jnp.zeros((mini_batch_size, 1), dtype=jnp.bool),
                ),
                axis=0,
            ),
        )

    def get_templates(self) -> tuple[Features, Names]:
        """Return a matrix of templates created from dataset.

        Randomly pulls `self.template_size` samples for each label, averages
        them together, and returns the templates. Use `self.label_names` to get
        the corresponding names.
        """
        ts = self.template_size
        frequent_labels = self.labels.sum(0) > ts
        index = self._rng.permuted(
            self.labels[:, frequent_labels]
            * (np.arange(self.labels.shape[0]) + 1).reshape((-1, 1)),
            axis=0,
        )
        col_indices = (
            index[index[:, i] > 0, i] - 1 for i in range(index.shape[1])
        )

        dtype = np.dtype((self.features.dtype, (ts, self.features.shape[1])))
        return (
            jnp.asarray(
                np.fromiter(
                    (self.features[idx[:ts], :] for idx in col_indices),
                    count=self.labels.shape[1],
                    dtype=dtype,
                ).mean(axis=1)
            ),
            self.label_names[frequent_labels],
        )


def from_huggingface(
    dataset: datasets.Dataset,
    features: str = "embedding",
    sample_id="pmid",
    labels="gene2pubtator",
    **kwds,
) -> DataLoader:
    """Construct a DataLoader from a datasets.Dataset.

    Note only the actual data (features, labels, and names) are saved. Other
    DataLoader parameters must be set again (such as batch_size, template_size,
    splits, and random seed) using the kwds argument.

    Parameters
    ----------
    dataset : dataset.DataSet
        The dataset to convert.
    features : str, default "embedding"
        The name of the dataset feature to use as features.
    sample_id : str, default "pmid"
        The name of the dataset feature to use as IDs for the samples.
    labels : str, default "gene2pubtator"
        The name of the dataset feature to use as labels. This should have the
        dataset.Feature type dataset.ClassLabel. This will be used for both
        label values and their symbols. In addition to "gene2pubtator",
        "gene2pubmed" is a useful option.
    **kwds :
        All other keywords are passed to the DataLoader constructor.

    Return
    ------
    dataloader : DataLoader
        The new DataLoader.

    """

    def to_sparse_labels(
        dataset: datasets.Dataset, labels: str
    ) -> sp.sparse.csc_array:
        label_list = dataset[labels]
        shape = (len(label_list), len(dataset.features[labels].feature.names))
        coords = np.fromiter(
            (
                (i, lab_id)
                for i, ids in enumerate(label_list)
                for lab_id in ids
            ),
            dtype=(np.dtype(np.int64), 2),
        )
        data = np.ones((coords.shape[0],), dtype=np.bool)
        return sp.sparse.coo_array((data, coords.T), shape).tocsc()

    return DataLoader(
        jnp.asarray(dataset[features]),
        to_sparse_labels(dataset, labels),
        dataset[sample_id],
        dataset.features[labels].feature.names,
        **kwds,
    )


# TODO: Once the embeddings dataset is upload to huggingface, switch from
# datasets.load_from_disk to datasets.load_dataset and let hf handle all
# caching,
def load_dataset(
    name: str,
    data_dir: str | None = None,
    labels: str = "gene2pubtator",
    **kwds,
):
    """Load a dataset.Dataset and convert it to a DataLoader.

    Wrapper around `dataset.load_dataset` and `from_huggingface`. For more
    control (such as filtering the dataset before sending to the DataLoader),
    use `dataset.load_dataset` and preprocess the dataset, then call
    `abstract2gene.dataset.from_huggingface`. This also does not expose all
    options availble to `from_huggingface`, for more flexibility call directly.

    Parameters
    ----------
    name : str
        Name of the data set to read.
    data_dir : Optional str
        Where to search for the dataset. Defaults to
        `abstract2gene.storage.default_data_dir` with "datasets" subdir
        appended to it.
    labels : str, default "gene2pubtator"
        The name of the dataset feature to use for labels. Feature must have
        type dataset.ClassLabel.
    **kwds :
        Keyword arguments to be forwarded to the DataLoader constructor.

    Returns
    -------
    dataset : DataLoader

    """
    path = os.path.join(data_dir or default_data_dir("datasets"), name)
    return from_huggingface(
        datasets.load_from_disk(path), labels=labels, **kwds
    )
