__all__ = ["Trainer", "test"]

import datasets
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx

from abstract2gene.dataset import DataLoaderDict
from abstract2gene.typing import Batch, Samples

from ._models import Model
from ._utils import MultiLabelAccuracy, eval_fn, train_fn


class Trainer:
    def __init__(
        self,
        model: Model,
        data: DataLoaderDict,
        tx: optax.GradientTransformation,
    ):
        self.model = model
        self.data = data
        self.optimizer = nnx.Optimizer(model, tx)
        # Define here so we can resume training.
        self.epoch = 0
        self.metrics = nnx.MultiMetric(
            accuracy=MultiLabelAccuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

    def _compute_templates(self, templates: Samples) -> Samples:
        return self.data.fold_templates(self.model(templates)).mean(axis=1)

    def train_step(self, batch: Batch) -> None:
        templates, x = self.data.split_batch(batch[0])
        templates = self._compute_templates(templates)
        loss, preds, grads = train_fn(self.model, x, templates, batch[-1])
        self.metrics.update(loss=loss, predictions=preds, labels=batch[-1])
        self.optimizer.update(grads)

    def eval_step(self, batch: Batch) -> None:
        templates, x = self.data.split_batch(batch[0])
        templates = self._compute_templates(templates)
        loss, preds = eval_fn(self.model, x, templates, batch[-1])
        self.metrics.update(loss=loss, predictions=preds, labels=batch[-1])

    def write_metrics(self, stage: str, step: int):
        for metric, value in self.metrics.compute().items():
            self.history[f"{stage}_{metric}"].append(value)  # type: ignore[arg-type]

        self.metrics.reset()

    def train(
        self,
        max_epochs: int = 100,
        stop_delta: float = 1e-5,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train weights for the model."""

        def validate_epoch():
            for batch in self.data["validate"].batch():
                self.eval_step(batch)

            self.write_metrics("test", epoch)

        def train_epoch():
            for batch in self.data["train"].batch():
                self.train_step(batch)

            self.write_metrics("train", epoch)

        self.model.train()

        window_size = 20
        delta = np.full(window_size, np.nan)
        delta[0] = stop_delta + 1
        last_err = 0.0
        start_epoch = self.epoch
        for epoch in range(start_epoch, max_epochs + start_epoch):
            if abs(np.nanmean(delta)) < stop_delta:
                break

            try:
                train_epoch()
                validate_epoch()
            except KeyboardInterrupt:
                # End training gracefully on keyboard interrupt. Model will
                # still be trained.
                print("\nExiting training loop")
                break

            self.epoch += 1
            err = self.history["test_loss"][-1]
            if epoch > start_epoch:
                delta[epoch % window_size] = last_err - err

            last_err = err

            if verbose:
                train_err = self.history["train_loss"][-1]
                train_acc = self.history["train_accuracy"][-1]
                test_acc = self.history["test_accuracy"][-1]

                print(f"Epoch: {epoch}")
                print(f"  Train loss: {train_err:.4g}")
                print(f"  Test loss: {err:.4g}")
                print(f"  Train accuracy: {train_acc:.4g}")
                print(f"  Test accuracy: {test_acc:.4g}")
                print(f"  Delta: {delta[epoch % window_size]:.4g}")
                print(f"  Avg delta: {delta.mean():.4g}")

        return self.history


def test(
    model: Model,
    dataset: datasets.Dataset,
    label_name: str,
    symbols: list[str] | None = None,
    batch_size: int | None = None,
    n_samples: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a model on samples from dataset and compare to ground truth.

    Parameters
    ----------
    model : Model,
        The already trained model to run.
    dataset : datasets.Dataset
        A testing dataset.
    label_name : str
        The name of the dataset feature to use as labels.
    symbols : list[str], Optional
        A list of symbols, should have one symbol per label. If not provided
        the ClassLabel objects names will be used.
    batch_size: int, Optional
        If provided how many samples to run on the model at once. If not given,
        run all samples together.
    n_samples : int, Optional
        The number of samples to run.
    seed : int, Optional
        A random seed used for reproducible results.

    Returns
    -------
    results : pd.Dataframe
        A dataframe containing the predictions for a random sample of 20
        labels.

    """

    def multihot(labels: list[int]) -> np.ndarray:
        """Convert a list of integers to a multihot array.

        Uses the template indices to sync with output templates.
        """
        return np.isin(label_indices, labels)

    if model.templates is None:
        raise ValueError("Templates most be attached to model before testing.")

    model.eval()
    label_indices = model.sync_indices(dataset)

    rng = np.random.default_rng(seed=seed)

    n_samples = n_samples or len(dataset)
    n_samples = min(n_samples, len(dataset))
    batch_size = batch_size or n_samples
    n_batches = n_samples // batch_size

    batch_regression = []

    mini_dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    abstracts = mini_dataset["abstract"]
    titles = mini_dataset["title"]
    for i in range(n_batches):
        start = i * batch_size
        samples = [
            title + "[SEP]" + abstract
            for title, abstract in zip(
                titles[start : (start + batch_size)],
                abstracts[start : (start + batch_size)],
            )
        ]

        batch_regression.append(model.predict(samples)[:, label_indices >= 0])

    label_indices = label_indices[label_indices >= 0]
    regression = np.vstack(batch_regression)
    labels = np.vstack([multihot(labs) for labs in mini_dataset[label_name]])
    sample_pmids = np.asarray(mini_dataset["pmid"])

    preds = regression > 0.5

    tp = np.logical_and(preds, labels).sum()
    tn = np.logical_and(np.logical_not(preds), np.logical_not(labels)).sum()
    fp = np.logical_and(preds, np.logical_not(labels)).sum()
    fn = np.logical_and(np.logical_not(preds), labels).sum()

    binary_acc = 0
    n_examples = labels.sum(axis=0)
    for i in range(labels.shape[1]):
        binary_acc += preds[labels[:, i], i].sum()
        negatives = rng.choice(
            preds[np.logical_not(labels[:, i]), i],
            n_examples[i],
            replace=False,
        )
        binary_acc += (negatives == 0).sum()

    binary_acc /= 2 * n_examples.sum()

    print(f"accuracy: {(tp + tn) / preds.size}")
    print(f"sensitivity: {tp / (tp + fp)}")
    print(f"specificity: {tn / (tn + fn)}")
    print(f"Bin ACC: {binary_acc}\n")

    n_samples = 20
    label_mask = labels.sum(axis=0) >= n_samples
    n_labels = min(20, label_mask.sum())

    labels = labels[:, label_mask].astype(np.bool)
    regression = regression[:, label_mask]
    symbols = symbols or dataset.features[label_name].feature.names
    symbols = [symbols[int(idx)] for idx in label_indices[label_mask]]

    selected = rng.choice(labels.shape[1], n_labels)

    scores = np.zeros((2 * n_labels * n_samples,))
    pmids = np.zeros((2 * n_labels * n_samples,), like=sample_pmids)
    tags = np.tile(
        np.concat(
            (
                np.asarray(["match"]).repeat(n_samples),
                np.asarray(["differ"]).repeat(n_samples),
            )
        ),
        n_labels,
    )

    label_names = (
        np.asarray([symbols[idx] for idx in selected])
        .reshape((-1, 1))
        .repeat(2 * n_samples)
    )

    last = 0
    for label in selected:
        match = regression[labels[:, label], label]
        pmid_match = sample_pmids[labels[:, label]]
        differ = regression[jnp.logical_not(labels[:, label]), label]
        pmid_differ = sample_pmids[jnp.logical_not(labels[:, label])]
        idx = rng.permuted(np.arange(len(match)))[:n_samples]
        scores[last : (last + n_samples)] = match[idx]
        pmids[last : (last + n_samples)] = pmid_match[idx]
        last += n_samples
        idx = rng.permuted(np.arange(len(match)))[:n_samples]
        scores[last : (last + n_samples)] = differ[idx]
        pmids[last : (last + n_samples)] = pmid_differ[idx]
        last += n_samples

    return pd.DataFrame(
        {"pmid": pmids, "score": scores, "tag": tags, "symbol": label_names}
    )
