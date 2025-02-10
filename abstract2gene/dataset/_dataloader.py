"""Collect feature data and provide iterable sets."""

from __future__ import annotations

__all__ = [
    "DataLoader",
    "DataLoaderDict",
    "from_huggingface",
    "mock_dataloader",
]

from collections import UserDict
from typing import Iterable, Sequence, TypeAlias

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from ..typing import Batch, Samples

InLabels: TypeAlias = sp.sparse.csc_array


class DataLoaderDict(UserDict):
    def __init__(
        self,
        mapping: dict[str, DataLoader],
        rng: np.random.Generator,
        batch_size: int = 64,
        template_size: int = 32,
        labels_per_batch: int = -1,
        max_steps: int = -1,
    ):
        """Initialize a DataLodareDict.

        Parameters
        ----------
        mapping : dict[str, DataLoader]
            The dictionary of DataLoaders to handle.
        rng : numpy random generator
            A random generator.
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
        max_steps : int, Optional
            If set, this limits the number of batches that will be returned in
            a single epoch.

        """
        super().__init__(mapping)
        self._rng = rng
        self._batch_size = batch_size
        self._template_size = template_size
        self.labels_per_batch = labels_per_batch
        self.update_params()
        self.n_features = mapping[list(mapping.keys())[0]].n_features
        self.n_samples = mapping[list(mapping.keys())[0]].n_samples
        self.max_steps = max_steps

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

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @max_steps.setter
    def max_steps(self, n: int):
        self._max_steps = n
        for dl in self.data.values():
            dl.max_steps = n

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

    def get_templates(
        self, template_size: int | None = None
    ) -> tuple[jax.Array, np.ndarray]:
        """Pull samples at random for creating templates.

        If template_size is set this many samples will be pulled for each
        template. Otherwise use self.template_size.

        In most cases this function should be used indirectly by supplying a
        DataLoaderDict to a model's attach_templates function.

        Returns
        -------
        template_samples : jax.Array
            A 2d Array of size (n_templates x template_size) x n_features. The
            number of templates is a function of the number of labels with at
            least template_size samples.
        label_indices : np.ndarray
            A list of the label indices associated with each template in
            template_samples. This (and by extension template_samples) will be
            sorted in ascending order. If there's enough samples for each label
            this will be 0--n_labels. (Useful for associated with a HuggingFace
            dataset using the same indices as this loader's data.)

        See Also
        --------
        DataLoaderDict.fold_templates; to fold into a 3d Array.
        Model.attach_templates: uses this function to fix templates to a model.

        """
        template_size = template_size or self.template_size

        key = list(self.data.keys())[0]
        samples = self.data[key].samples

        labels = sp.sparse.hstack([dl._labels for dl in self.data.values()])
        indices = np.concat([dl.label_indices for dl in self.data.values()])

        sort_idx = np.argsort(indices)
        labels = labels[:, sort_idx]
        indices = indices[sort_idx]

        mask = labels.sum(axis=0) > template_size
        labels = labels[:, mask].toarray()
        indices = indices[mask]

        rows = np.arange(labels.shape[0])
        templates = np.vstack(
            [
                samples[self._rng.permuted(rows[labs])[:template_size], :]
                for labs in labels.T
            ],
            dtype=samples.dtype,
        )

        return (jnp.asarray(templates), indices)

    def split_batch(
        self, batch: Samples, template_size: int | None = None
    ) -> tuple[Samples, Samples]:
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
        template_size = template_size or self.template_size
        # Since self.labels_per_batch may be -1 (for all labels) can't use the
        # variable so calculate n labels instead.
        n_labels = (batch.shape[0] - self.batch_size) // template_size
        n_temp_rows = n_labels * template_size
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
    data = from_huggingface(HF_dataset)
    for batch in data.batch():
       train_step(model, batch)

    See Also:
    --------
    Suggested method for creating a dataset.
    `abstract2gene.datasets.bioc2dataset`

    """

    def __init__(
        self,
        samples: np.ndarray,
        labels: InLabels,
        label_indices: np.ndarray,
        seed: int = 0,
    ):
        """Construct a DataLoader.

        Parameters
        ----------
        samples : jax.Array
            Should be in the form n_samples x n_features
        labels : 2d sparse array (csc form)
            Should be in the form n_samples x n_labels.
        label_indices : np.ndarray
            The indices of the label.
        seed : int, default 0
            The seed for the random number generator.

        """
        self._samples = samples
        self._labels = labels
        self._template_mask: np.ndarray = np.asarray([])

        self._ext_label_indices = label_indices

        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._batch_sample_names: list[str] = []
        self._batch_label_names: list[str] = []
        self._batch_label_indices: np.ndarray = np.asarray([])

        self._label_idx: np.ndarray = np.asarray([])

        self._bs = 0
        self._ts = 0
        self._labels_per_batch = 0
        self.max_steps = 0

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

    @property
    def samples(self):
        return self._samples

    @property
    def labels(self) -> InLabels:
        return self._labels[:, self._label_idx]

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

    @property
    def label_indices(self) -> np.ndarray:
        return self._ext_label_indices

    @label_indices.setter
    def label_indices(self):
        raise RuntimeError("Label indices cannot be modified.")

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
        )
        return "[samples: {}, features: {}, labels: {}]".format(*data)

    def __getitem__(self, key) -> DataLoader:
        new = DataLoader(
            self.samples[key, :],
            self._labels[key, :],
            self.label_indices,
            self._seed,
        )
        new._update_params(self._bs, self._ts, self._labels_per_batch)
        return new

    def batch(self) -> Iterable[Batch]:
        """Generate batches of samples to train on."""
        label_pool = self._rng.permutation(self._label_idx)
        cut = self.labels_per_batch * (
            label_pool.shape[0] // self.labels_per_batch
        )
        label_pool = label_pool[:cut].reshape((-1, self.labels_per_batch))
        labels = self._labels

        batch_n = 0
        for batch_labels in label_pool:
            for batch in self._split_labels(labels[:, batch_labels]):
                yield batch
                batch_n += 1

            if (self.max_steps > 0) and (batch_n > self.max_steps):
                break

    def _split_labels(
        self,
        labels: sp.sparse.sparray,
    ) -> Iterable[Batch]:
        ts = self.template_size
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

        percent_true = 0.8
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

            yield (
                jnp.concat(
                    (
                        templates,
                        *tuple(self.samples[draw, :] for draw in draws),
                    )
                ),
                jnp.concat(tuple(labels[draw, :].todense() for draw in draws)),
            )


def from_huggingface(
    dataset: datasets.Dataset,
    samples: str = "embedding",
    labels: str = "gene",
    max_sample_labels: int = 10,
    split: dict[str, float] = {"train": 0.9, "validate": 0.1},
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
    labels : str, default "gene2pubtator"
        The name of the dataset feature to use as labels. This should have the
        dataset.Feature type dataset.ClassLabel. This will be used for both
        label values and their symbols. In addition to "gene2pubtator",
        "gene2pubmed" is a useful option.
    max_sample_labels : int, default 10
        Drop samples with more than `max_sample_labels` labels. With too many
        labels, the sample cannot be specific to a single label and therefore
        is expected to not be labeled well.
    split : dict[str, float]
        Key-value pairs of proportions for each split. Sum of proportions
        should equal 1.
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

        return {k: mask == i + 1 for i, k in enumerate(split)}

    rng: np.random.Generator = np.random.default_rng(seed)
    new_seed = rng.integers(9999, size=len(split)).astype(int)
    splabels = to_sparse_labels(dataset, labels)
    label_masks = split_labels(splabels, split, rng)
    feats = dataset.with_format("numpy", columns=[samples])[samples]

    over_labeled = splabels.sum(axis=1) > max_sample_labels
    splabels = splabels[np.logical_not(over_labeled), :]
    feats = feats[np.logical_not(over_labeled), :]
    labeled = splabels.sum(axis=1) > 0

    unlabeled = None
    if return_unlabeled:
        unlabeled = feats[np.logical_not(labeled)]

    feats = feats[labeled, :]
    splabels = splabels[labeled, :]

    dataloaders = DataLoaderDict(
        {
            k: DataLoader(
                feats,
                splabels[:, label_masks[k]],
                np.arange(splabels.shape[1])[label_masks[k]],
                seed=sd,
            )
            for sd, k in zip(new_seed, split)
        },
        rng=rng,
        **kwds,
    )

    return (dataloaders, unlabeled)


def mock_dataloader(
    n_samples: int = 1000,
    n_features: int = 40,
    n_classes: int = 20,
    noise: float = 0.5,
    seed: int = 0,
) -> DataLoaderDict:
    rng = np.random.default_rng(seed)
    labels = sp.sparse.coo_array(
        (
            np.ones((n_samples,), dtype=np.bool),
            (np.arange(n_samples), [i % n_classes for i in range(n_samples)]),
        )
    ).tocsc()

    samples: np.ndarray | jax.Array = rng.normal(
        noise, 1, size=(n_samples, n_features)
    )

    def class_number(labels, index):
        pos = np.diff(labels[[index], :].indptr)
        return np.where(pos)

    for samp in np.arange(n_samples):
        label = class_number(labels, samp)
        samples[samp, label] += 5

    idx = rng.permuted(np.arange(n_samples))
    samples = np.asarray(samples[idx, :])
    labels = labels[idx, :]
    split: dict = {"train": 0.6, "test": 0.2, "validate": 0.2}
    pos = 0
    for k, v in split.items():
        split[k] = [pos, pos + int(v * n_classes)]
        pos += int(v * n_classes)

    split["validate"][1] = n_classes

    return DataLoaderDict(
        {
            k: DataLoader(
                samples,
                labels[:, v[0] : v[1]],
                0,
            )
            for k, v in split.items()
        },
        batch_size=20,
        template_size=16,
        labels_per_batch=4,
    )
