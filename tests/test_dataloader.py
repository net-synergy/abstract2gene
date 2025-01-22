import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp

from abstract2gene.dataset import DataLoader, DataLoaderDict


class TestDataLoader:
    n_labels = 50
    labels_per_batch = 6
    samples_per_label = 20
    n_samples = n_labels * samples_per_label
    n_feats = 2
    batch_size = 5
    template_size = 10

    def _generate_dataloader(self) -> DataLoaderDict:
        sample_nums = (
            jnp.arange(self.samples_per_label)
            .reshape((-1, 1))
            .repeat(self.n_labels, axis=1)
            .T.reshape((-1))
        )
        label_nums = jnp.arange(self.n_labels).repeat(self.samples_per_label)
        samples = jnp.stack((sample_nums, label_nums), axis=1)
        labels = sp.sparse.coo_array(
            (
                np.ones((samples.shape[0]), dtype=np.bool),
                (np.arange(samples.shape[0]), samples[:, 1]),
            )
        ).tocsc()

        sample_ids = [str(idx) for idx in range(self.n_samples)]
        label_ids = [str(idx) for idx in range(self.n_labels)]

        dl = DataLoader(samples, labels, sample_ids, label_ids, 0)
        return DataLoaderDict(
            {"train": dl},
            batch_size=self.batch_size,
            template_size=self.template_size,
            labels_per_batch=self.labels_per_batch,
        )

    def test_n_labels(self):
        dl = self._generate_dataloader()["train"]
        assert dl.n_labels == self.n_labels
        new_labels = dl._labels.tolil()
        new_labels[:, [1]] = 0
        dl._labels = new_labels.tocsc()
        dl._update_params(
            self.batch_size, self.template_size, self.labels_per_batch
        )
        assert dl.n_labels == (self.n_labels - 1)

    def test_batch_template_folding(self):
        dld = self._generate_dataloader()
        batch = next(dld["train"].batch())
        templates, _ = dld.split_batch(*batch[:-1])
        templates = dld.fold_templates(templates)

        expected_shape = (
            self.labels_per_batch,
            self.template_size,
            self.n_feats,
        )
        assert all(
            temp_shape == expected
            for temp_shape, expected in zip(templates.shape, expected_shape)
        )

        for i in range(templates.shape[0]):
            expected = templates[i, 0, 1]
            assert all(
                templates[i, j, 1] == expected
                for j in range(templates.shape[1])
            )

        templates = templates.mean(axis=1)

    def test_eval_template_folding(self):
        dld = self._generate_dataloader()
        dld.update_params(labels_per_batch=-1)
        dl = dld["train"]
        dl.eval()

        assert dl.n_samples == self.n_samples - (
            self.n_labels * self.template_size
        )

        templates = dld.fold_templates(dl.templates)
        expected_shape = (
            self.n_labels,
            self.template_size,
            self.n_feats,
        )
        assert all(
            temp_shape == expected
            for temp_shape, expected in zip(templates.shape, expected_shape)
        )

        for i in range(templates.shape[0]):
            expected = templates[i, 0, 1]
            assert all(
                templates[i, j, 1] == expected
                for j in range(templates.shape[1])
            )

        samples = dl.samples
        for label in range(self.n_labels):
            labels_samples = jnp.concat(
                (
                    templates[label, :, 0],
                    samples[dl.labels[:, [label]].toarray().squeeze(), 0],
                )
            )
            assert jnp.all(
                jnp.sort(labels_samples) == jnp.arange(self.samples_per_label)
            )
