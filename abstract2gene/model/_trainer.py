__all__ = ["Trainer"]

import os
from datetime import datetime

import numpy as np
import optax
from flax import nnx
from flax.metrics import tensorboard

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

        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        log_dir = os.path.join("./tmp", model.name, timestamp)
        self.writers = {
            k: tensorboard.SummaryWriter(os.path.join(log_dir, k))
            for k in ["train", "test"]
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
            self.writers[stage].scalar(metric, value, step)

        self.metrics.reset()
        self.writers[stage].flush()

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

        self.data["train"].train()
        self.data["validate"].train()

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

    def test(self):
        pass
