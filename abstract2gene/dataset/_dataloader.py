"""Collect feature data and provide iterable sets."""

from __future__ import annotations

__all__ = [
    "DataLoader",
    "DataLoaderDict",
    "from_huggingface",
    "load_dataset",
]

import os
from collections import UserDict
from typing import Iterable, Sequence, TypeAlias

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from abstract2gene.storage import default_data_dir

from ..typing import Batch, Names, Samples

InLabels: TypeAlias = sp.sparse.csc_array


class DataLoaderDict(UserDict):
    def __init__(
        self,
        mapping: dict[str, DataLoader],
        batch_size: int = 64,
        template_size: int = 32,
        labels_per_batch: int = -1,
    ):
        """Initialize a DataLodareDict.

        Parameters
        ----------
        mapping : dict[str, DataLoader]
            The dictionary of DataLoaders to handle.
        batch_size : int, default 64
            How many samples should make up a batch.
        template_size : int, default 32
            The number of samples averaged together to make a template.
        labels_per_batch : int, Optional
            How many labels to use when making a batch. By default uses all
            labels. If less than the total number of labels, batches will be
            made from a pool of samples labeled with at least one of the
            randomly selected `labels_per_batch` labels (plus ~10% of a batches
            samples will not include one of the batch labels). For example, if
            `labels_per_batch` is 4, 4 labels will be randomly selected,
            batches will be made of samples with these labels until the pool
            runs dry. The resulting batch labels will have 4 columns.

        """
        super().__init__(mapping)
        self._batch_size = batch_size
        self._template_size = template_size
        self.labels_per_batch = labels_per_batch
        self.update_params()
        self.n_features = mapping[list(mapping.keys())[0]].n_features
        self.n_samples = mapping[list(mapping.keys())[0]].n_samples

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs: int):
        self._batch_size = bs
        self.update_params()

    @property
    def template_size(self) -> int:
        return self._template_size

    @template_size.setter
    def template_size(self, ts: int):
        self._template_size = ts
        self.update_params()

    @property
    def labels_per_batch(self) -> int:
        return self._labels_per_batch

    @labels_per_batch.setter
    def labels_per_batch(self, n: int):
        self._labels_per_batch = n
        self.update_params()

    def __repr__(self) -> str:
        data = (self.batch_size, self.template_size, self.labels_per_batch)
        out = "batch size: {}, template size: {}, labels per batch: {}"
        out = out.format(*data) + "\n  {"
        max_len = max((len(k) for k in self.data))
        for k, v in self.data.items():
            spacer = " " * (max_len - len(k))
            out += f"\n    {spacer}{k}: {v}"

        out += "\n  }"
        return out

    def __getitem__(self, key) -> DataLoader:
        return super().__getitem__(key)

    def select(self, indices: Sequence[int]) -> DataLoaderDict:
        return DataLoaderDict(
            {k: v[indices] for k, v in self.data.items()},
            batch_size=self.batch_size,
            template_size=self.template_size,
            labels_per_batch=self.labels_per_batch,
        )

    def update_params(
        self,
        batch_size: int | None = None,
        template_size: int | None = None,
        labels_per_batch: int | None = None,
    ):
        batch_size = batch_size or self.batch_size
        template_size = template_size or self.template_size
        labels_per_batch = labels_per_batch or self.labels_per_batch
        {
            dl._update_params(batch_size, template_size, labels_per_batch)
            for dl in self.data.values()
        }
        self._batch_size = batch_size
        self._template_size = template_size
        self._labels_per_batch = labels_per_batch

    def reset_rngs(self):
        for dl in self.data.values():
            dl.reset_rng()

    def split_batch(self, batch: Samples) -> tuple[Samples, Samples]:
        """Split the samples of a batch into templates and samples.

        The first `labels_per_batch * template_size` columns of a batch's
        samples are the templates. This splits the batch's samples into the
        templates and the actual training or testing samples. Templates are
        returned as an array of (labels_per_batch * template_size) x n_features

        See fold_templates to convert to a 3d array of labels x template_size x
        features.

        Returns
        -------
        templates : jax.Array
            The templates for each label.
        samples : jax.Array
            The samples.

        """
        # Since self.labels_per_batch may be -1 (for all labels) can't use the
        # variable so calculate n labels instead.
        n_labels = (batch.shape[0] - self.batch_size) // self.template_size
        n_temp_rows = n_labels * self.template_size
        templates = batch[:n_temp_rows, :]
        samples = batch[n_temp_rows:, :]
        return (templates, samples)

    def fold_templates(self, templates: Samples) -> Samples:
        """Fold templates into a 3d array.

        Returned templates are n_labels x template_size x features. Reduce over
        axis 1 to get a single template per label.
        """
        n_labels = templates.shape[0] // self.template_size
        return templates.reshape((n_labels, self.template_size, -1))


class DataLoader:
    """Collects samples and labels to load batches.

    Example:
    -------
    data = load_dataset("path/to/dataset")
    for batch in data.batch():
       train_step(model, batch)

    See Also:
    --------
    Suggested method for creating a dataset.
    `abstract2gene.datasets.bioc2dataset`

    """

    def __init__(
        self,
        samples: Samples,
        labels: InLabels,
        sample_names: Names,
        label_ids: Names,
        label_symbols: Names,
        seed: int = 0,
    ):
        """Construct a DataLoader.

        Parameters
        ----------
        samples : jax.Array
            Should be in the form n_samples x n_features
        labels : 2d sparse array (csc form)
            Should be in the form n_samples x n_labels.
        sample_names : list[str]
            One dimensional array of feature names (such as PMIDs)
        label_ids, label_symbols : list[str]
            Names for the label columns. IDs should be unique and consistent
            (such as NCBI Gene IDs). Symbols can be more flexible. The symbols
            are intended to be meaningful to a human whereas IDs are intended
            to be used to link between datasets.
        seed : int, default 0
            The seed for the random number generator.

        """
        self._is_training = True

        self._samples = samples
        self._sample_names = sample_names
        self._labels = labels
        self._label_ids = label_ids
        self._label_symbols = label_symbols
        self._template_mask: np.ndarray = np.asarray([])

        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._batch_sample_names: list[str] = []
        self._batch_label_names: list[str] = []

        self._label_idx: np.ndarray = np.asarray([])

        self._bs = 0
        self._ts = 0
        self._labels_per_batch = 0

    def _update_params(
        self, batch_size: int, template_size: int, labels_per_batch: int
    ) -> None:
        self._bs = batch_size
        self._ts = template_size
        self._labels_per_batch = labels_per_batch
        self._all_labels = labels_per_batch <= 0

        if self._all_labels:
            min_occurance = self.template_size
        else:
            min_occurance = (
                self.batch_size // self.labels_per_batch
            ) + self.template_size

        self._label_idx = np.arange(self._labels.shape[1])[
            self._labels.sum(axis=0) > min_occurance
        ]

        if self._all_labels:
            self._labels_per_batch = self.n_labels

        # If template size or labels per batch change, this will change the
        # shape of the templates so they need to be recalculated.
        if not self.is_training:
            self.eval()

    @property
    def is_training(self) -> bool:
        """Whether the dataset is in training mode or eval mode."""
        return self._is_training

    @property
    def samples(self):
        if self.is_training:
            return self._samples

        return self._samples[self._sample_mask, :]

    @property
    def sample_names(self):
        if self.is_training:
            return self._sample_names

        return [
            name
            for mask, name in zip(self._sample_mask, self._sample_names)
            if mask
        ]

    @property
    def templates(self) -> Samples:
        if not self.is_training:
            # Most not be none if in eval mode.
            return self._samples[self._template_idx, :]

        raise RuntimeError("Dataset not in evaluation mode.")

    @property
    def labels(self) -> InLabels:
        if self.is_training:
            labels = self._labels
        else:
            labels = self._labels[self._sample_mask, :]

        return labels[:, self._label_idx]

    @property
    def label_ids(self) -> Names:
        return [self._label_ids[i] for i in self._label_idx]

    @property
    def label_symbols(self) -> Names:
        return [self._label_symbols[i] for i in self._label_idx]

    @property
    def batch_size(self) -> int:
        return self._bs

    @batch_size.setter
    def batch_size(self, ts: int):
        raise RuntimeError(
            "Set the batch size through the DataLoaderDict class"
        )

    @property
    def template_size(self) -> int:
        return self._ts

    @template_size.setter
    def template_size(self, ts: int):
        raise RuntimeError(
            "Set the template size through the DataLoaderDict class"
        )

    @property
    def labels_per_batch(self) -> int:
        return self._labels_per_batch

    @labels_per_batch.setter
    def labels_per_batch(self, ts: int):
        raise RuntimeError(
            "Set the labels per batch through the DataLoaderDict class"
        )

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the RNG to the original seed.

        If `seed` is provided use instead of the original seed.
        """
        seed = seed or self._seed
        self._rng = np.random.default_rng(seed)

    @property
    def n_features(self) -> int:
        """Return the dimensionality of the feature vector."""
        return self.samples.shape[1]

    @property
    def n_labels(self) -> int:
        """Return number of unique labels.

        Note this is the number of labels with at least batch_size +
        template_size samples not the raw number of columns in the label
        matrix. As such this value can change if batch size, template size, or
        labels_per_batch are reset.
        """
        return len(self._label_idx)

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return self.samples.shape[0]

    def __repr__(self) -> str:
        data = (
            self.n_samples,
            self.n_features,
            self.n_labels,
            "train" if self.is_training else "eval",
        )
        return "[samples: {}, features: {}, labels: {}, mode: {}]".format(
            *data
        )

    def __getitem__(self, key) -> DataLoader:
        new = DataLoader(
            self.samples[key, :],
            self._labels[key, :],
            self.sample_names[key],
            self._label_ids,
            self._label_symbols,
            self._seed,
        )
        new._update_params(self._bs, self._ts, self._labels_per_batch)
        return new

    def batch_label_name(self) -> Names:
        """Return the current batch's label name."""
        return self._batch_label_names

    def batch_sample_names(self) -> Names:
        """Return the names for the batch's samples."""
        return self._batch_sample_names

    def batch(self) -> Iterable[Batch]:
        """Generate batches of samples to train on."""
        label_pool = self._rng.permutation(self._label_idx)
        cut = self.labels_per_batch * (
            label_pool.shape[0] // self.labels_per_batch
        )
        label_pool = label_pool[:cut].reshape((-1, self.labels_per_batch))
        if self.is_training:
            labels = self._labels
        else:
            labels = self._labels[self._sample_mask, :]
        for batch_labels in label_pool:
            self._batch_label_names = [
                self._label_symbols[idx] for idx in batch_labels
            ]
            for batch in self._split_labels(labels[:, batch_labels]):
                yield batch

    def _split_labels(
        self,
        labels: sp.sparse.sparray,
    ) -> Iterable[Batch]:
        ts = self.template_size if self.is_training else 0
        bs = self.batch_size

        label_samples = [
            self._rng.permutation(labels[:, [idx]].indices)
            for idx in range(labels.shape[1])
        ]

        templates = self.samples[
            np.concat(tuple(samps[:ts] for samps in label_samples)), :
        ]

        rows = np.arange(labels.shape[0])
        others_mask = labels.sum(axis=1) == 0
        samples = [
            self._rng.permutation(rows[others_mask]),
            self._rng.permutation(
                np.concat(tuple(samps[ts:] for samps in label_samples))
            ),
        ]

        # If we're using all labels, don't try to find examples without one of
        # the selected labels.
        percent_true = 0.8 if not self._all_labels else 1
        n_draw = [0, max(int(bs * percent_true), 1)]
        n_draw[False] = bs - n_draw[True]

        available = [len(pool) for pool in samples]
        ptr = [0, 0]

        labels = labels.astype(jnp.float32)
        while all(
            (p + n) < avail for p, n, avail in zip(ptr, n_draw, available)
        ):
            draws = [
                pool[p : p + n] for pool, p, n in zip(samples, ptr, n_draw)
            ]
            ptr = [p + n for p, n in zip(ptr, n_draw)]
            self._batch_sample_names = [
                self.sample_names[draw] for draw in np.concat(draws)
            ]

            yield (
                jnp.concat(
                    (
                        templates,
                        *tuple(self.samples[draw, :] for draw in draws),
                    )
                ),
                jnp.concat(tuple(labels[draw, :].todense() for draw in draws)),
            )

    def eval(self):
        """Set the datasets mode to eval.

        In eval mode, the dataset splits out a preselected set of samples for
        each template and places them in the templates parameter.

        The template samples are removed from the samples parameter and `batch`
        will not return templates along with samples.

        Use `DataLoader.train` to reverse this process.
        Use `DataLoader.is_training` to determine the current mode.

        """
        ts = self.template_size

        labels = self._labels[:, self._label_idx]
        n_samples = labels.sum(axis=0)
        training_mask = labels
        ones = np.ones((ts,), dtype=np.bool)
        for idx in range(training_mask.shape[1]):
            zeros = np.zeros((n_samples[idx] - ts,), dtype=np.bool)
            mask = self._rng.permutation(np.concat((ones, zeros)))
            labeled = labels[:, [idx]].toarray().squeeze()
            training_mask[labeled, idx] = mask

        samples = np.arange(labels.shape[0])
        self._template_idx = np.fromiter(
            (
                idx
                for col in range(training_mask.shape[1])
                for idx in samples[training_mask[:, [col]].toarray().squeeze()]
            ),
            dtype=np.dtype(int),
            count=(ts * training_mask.shape[1]),
        )
        self._sample_mask = np.logical_not(training_mask.sum(axis=1))
        self._is_training = False

    def train(self):
        self._is_training = True


def from_huggingface(
    dataset: datasets.Dataset,
    samples: str = "embedding",
    sample_id: str = "pmid",
    labels: str = "gene2pubtator",
    symbols: list[str] = [],
    max_sample_labels: int = 10,
    split: dict[str, float] = {"train": 0.8, "test": 0.1, "validate": 0.1},
    seed: int = 0,
    return_unlabeled: bool = False,
    **kwds,
) -> tuple[DataLoaderDict, jax.Array | None]:
    """Construct a set of DataLoaders from a datasets.Dataset.

    Creates a DataLoader for each key in `split`.

    Parameters
    ----------
    dataset : dataset.DataSet
        The dataset to convert.
    samples : str, default "embedding"
        The name of the dataset feature to use as samples.
    sample_id : str, default "pmid"
        The name of the dataset feature to use as IDs for the samples.
    labels : str, default "gene2pubtator"
        The name of the dataset feature to use as labels. This should have the
        dataset.Feature type dataset.ClassLabel. This will be used for both
        label values and their symbols. In addition to "gene2pubtator",
        "gene2pubmed" is a useful option.
    symbols : list[str]
        TEMPORARY A list of the label's symbols. Should be better integrated
        into the huggingface dataset.
    max_sample_labels : int, default 10
        Drop samples with more than `max_sample_labels` labels. With too many
        labels, the sample cannot be specific to a single label and therefore
        is expected to not be labeled well.
    split : dict[str, float]
        Key-value pairs of
    seed : int, 0
        Random seed to initialize RNG used to split labels and generate a seed
        to pass to the DataLoader.
    return_unlabeled : bool, default False
        Whether to return the unlabeled samples as an array.
    **kwds :
        All other keywords are passed to the DataLoader constructor.

    Return
    ------
    dataloader : dict[str, DataLoader]
        A dictionary of DataLoaders for each key in split.
    unlabeled_samples : jax.Array, optional
        If return_unlabeled is True, the array of all samples that had no
        labels is returned.

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

    def split_labels(
        labels: jax.Array, split: dict[str, float], rng: np.random.Generator
    ) -> dict[str, jax.Array]:
        """Reshuffle the training, test, and validation sets."""
        tol = 1e-5

        assert (sum(split.values()) > (1 - tol)) and (
            sum(split.values()) < (1 + tol)
        )

        label_mask = labels.sum(axis=0) > 0
        n_labels = sum(label_mask)

        szs = {k: int(n_labels * split[k]) for k, v in split.items()}
        szs[list(szs.keys())[-1]] = n_labels - sum(list(szs.values())[:-1])

        mask = np.zeros((labels.shape[1]))
        mask[label_mask] = np.concat(
            tuple(np.ones(sz) + i for i, sz in enumerate(szs.values()))
        )
        mask[label_mask] = rng.permutation(mask[label_mask])

        return {
            "train": mask == 1,
            "test": mask == 2,
            "validate": mask == 3,
        }

    rng: np.random.Generator = np.random.default_rng(seed)
    new_seed = rng.integers(9999, size=len(split)).astype(int)
    splabels = to_sparse_labels(dataset, labels)
    label_masks = split_labels(splabels, split, rng)
    feats = dataset.with_format("jax", columns=[samples])[samples]

    over_labeled = splabels.sum(axis=1) > max_sample_labels
    splabels = splabels[np.logical_not(over_labeled), :]
    feats = feats[np.logical_not(over_labeled), :]
    labeled = splabels.sum(axis=1) > 0

    unlabeled = None
    if return_unlabeled:
        unlabeled = feats[np.logical_not(labeled)]

    names = dataset.features[labels].feature.names
    # symbols = dataset.features[labels].feature.symbols
    feats = feats[labeled, :]
    splabels = splabels[labeled, :]

    sample_names = dataset[sample_id]
    if not isinstance(sample_names[0], str):
        sample_names = [str(name) for name in sample_names]

    dataloaders = DataLoaderDict(
        {
            k: DataLoader(
                feats,
                splabels[:, label_masks[k]],
                sample_names,
                [name for (mask, name) in zip(label_masks[k], names) if mask],
                [
                    symbol
                    for (mask, symbol) in zip(label_masks[k], symbols)
                    if mask
                ],
                seed=sd,
            )
            for sd, k in zip(new_seed, split)
        },
        **kwds,
    )

    return (dataloaders, unlabeled)


# TODO: Once the embeddings dataset is uploaded to huggingface, switch from
# datasets.load_from_disk to datasets.load_dataset and let hf handle all
# caching,
def load_dataset(
    name: str,
    data_dir: str | None = None,
    labels: str = "gene2pubtator",
    **kwds,
) -> tuple[DataLoaderDict, jax.Array | None]:
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
    dataloader : dict[str, DataLoader]
        A dictionary of DataLoaders for each key in split.
    unlabeled_samples : jax.Array, optional
        If return_unlabeled is True, the array of all samples that had no
        labels is returned.

    """
    import json

    path = os.path.join(data_dir or default_data_dir("datasets"), name)
    with open(os.path.join(path, "symbols.json"), "r") as f:
        symbols = json.load(f)

    return from_huggingface(
        datasets.load_from_disk(path), labels=labels, symbols=symbols, **kwds
    )
