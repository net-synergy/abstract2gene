__all__ = ["Trainer"]

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
        self,
        batch_size: int | None = None,
        seed: int = 0,
        threshold: float = 0.5,
    ) -> pd.DataFrame:

        batch_size_i = self.data.batch_size
        labels_per_batch_i = self.data.labels_per_batch
        self.data.update_params(batch_size=batch_size, labels_per_batch=-1)
        dataset = self.data["test"]

        self.model.eval()

        batch_labels = []
        batch_regression = []
        n_batches = 50
        for i, batch in enumerate(dataset.batch()):
            if i > n_batches:
                break
            templates, x = self.data.split_batch(batch[0])
            templates = self.data.fold_templates(self.model(templates)).mean(
                axis=1
            )
            batch_labels.append(batch[-1])
            batch_regression.append(self.model.predict(x, templates))

        labels = jnp.concat(batch_labels, axis=0)
        regression = jnp.concat(batch_regression, axis=0)

        preds = regression > threshold

        tp = jnp.logical_and(preds, labels).sum()
        tn = jnp.logical_and(
            jnp.logical_not(preds), np.logical_not(labels)
        ).sum()
        fp = jnp.logical_and(preds, np.logical_not(labels)).sum()
        fn = jnp.logical_and(np.logical_not(preds), labels).sum()

        print(f"accuracy: {(tp + tn) / preds.size}")
        print(f"sensitivity: {tp / (tp + fp)}")
        print(f"specificity: {tn / (tn + fn)}")

        n_labels = 20
        n_samples = 40
        label_mask = labels.sum(axis=0) > n_samples
        labels = labels[:, label_mask] == 1  # Convert from float to bool
        regression = regression[:, label_mask]
        label_names = [
            name
            for mask, name in zip(label_mask, dataset.batch_label_name())
            if mask
        ]

        rng = np.random.default_rng(seed=seed)
        selected = rng.choice(labels.shape[1], n_labels)

        scores = np.zeros((2 * n_labels * n_samples,))
        tags = np.tile(
            np.concat(
                (
                    np.asarray(["match"]).repeat(n_samples),
                    np.asarray(["differ"]).repeat(n_samples),
                )
            ),
            n_labels,
        )

        symbols = (
            np.asarray([label_names[idx] for idx in selected])
            .reshape((-1, 1))
            .repeat(2 * n_samples)
            .reshape((-1))
        )

        last = 0
        for label in selected:
            match = regression[labels[:, label], label]
            differ = regression[jnp.logical_not(labels[:, label]), label]
            scores[last : (last + n_samples)] = rng.permuted(match)[:n_samples]
            last += n_samples
            scores[last : (last + n_samples)] = rng.permuted(differ)[
                :n_samples
            ]
            last += n_samples

        self.data.update_params(
            batch_size=batch_size_i, labels_per_batch=labels_per_batch_i
        )
        return pd.DataFrame({"score": scores, "tag": tags, "symbol": symbols})

    def plot(self, df: pd.DataFrame, name: str | None = None):
        from plotnine import (
            aes,
            element_text,
            geom_errorbar,
            geom_point,
            ggplot,
            ggtitle,
            labs,
            position_dodge,
            position_jitterdodge,
            theme,
        )

        def stderr(x):
            return np.std(x) / np.sqrt(x.shape[0])

        metrics = df.groupby(["tag", "symbol"], as_index=False)["score"].agg(
            ["mean", "std", stderr]
        )

        jitter = position_jitterdodge(
            jitter_width=0.2, dodge_width=0.8, random_state=0
        )
        dodge = position_dodge(width=0.8)
        p = (
            ggplot(df, aes(x="symbol", color="tag"))
            + geom_point(aes(y="score"), position=jitter, size=1, alpha=0.4)
            + geom_errorbar(
                aes(
                    y="mean",
                    ymin="mean - (1.95 * std)",
                    ymax="mean + (1.95 * std)",
                    fill="tag",
                ),
                data=metrics,
                color="black",
                position=dodge,
                width=0.5,
                size=0.3,
            )
            + geom_errorbar(
                aes(
                    y="mean",
                    ymin="mean - (1.95 * stderr)",
                    ymax="mean + (1.95 * stderr)",
                    fill="tag",
                ),
                data=metrics,
                color="black",
                position=dodge,
                width=0.8,
                size=1,
            )
            + labs(y="Similarity", x="Gene")
            + ggtitle("Abstract embedding similarity")
            + theme(axis_text_x=element_text(angle=20))
        )
        if name:
            p.save(f"figures/model_comparison/{name}", width=10, height=10)
        else:
            p.show()
