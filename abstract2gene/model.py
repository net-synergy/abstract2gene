"""Models for comparing abstract features.

Various models for comparing abstract embeddings. This generally means
comparing an individual publication's abstract embeddings to template
embeddings for different labels (in the case of this package, genes).

Templates are the average of many examples of abstracts tagged with a label.
"""

__all__ = [
    "RawSimilarity",
    "SingleLayer",
    "MultiLayer",
    "MLPExtras",
    "Attention",
    "Trainer",
]

import os

import jax
import numpy as np
import optax
import pandas as pd
from flax import nnx
from flax.metrics import tensorboard

from .dataset import DataLoaderDict
from .typing import Batch, Labels, Names, Samples

RESULT_TEMPLATE = "results/{name}_validation.tsv"
RESULTS_TABLE = "results/model_comparison.tsv"


class Model(nnx.Module):
    """Base class for abstract2gene prediction models.

    Models can be trained in order to predict how similar an abstract's LLM
    embedding is to a template.

    Learns label independent weights. The weights are used to create new
    features that are a linear combination of the original features. Since the
    are not trained to the specific labels, they can be used to improve
    prediction on labels that haven't been seen during training.

    Use the model by calling the `predict` method (after training) or calling
    the model directly (equivalent).

    The model will return an array of predictions with length
    `templates.n_labels`. The labels of each prediction are in
    `self.label_names` such that `out[i]` is the prediction for
    `self.label_names[i]`.
    """

    def __init__(
        self,
        name: str = "",
    ):
        """Initialize a model.

        Parameters
        ----------
        name : str, optional
            A name to give to the model. This is only important for determining
            where to store test results and not needed for prediction.

        """
        self.name = name
        self.result_file = RESULT_TEMPLATE.format(name=name)
        self.templates: Samples | None = None
        self.label_names: Names | None = None

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        self.result_file = RESULT_TEMPLATE.format(name=value)

    def attach_templates(self, templates: Samples, names: Names) -> None:
        """Add the templates for the model.

        These are used for prediction---outside of training. During training,
        templates are created with batches by the dataset.

        Note the dataset should be of class `DataLoader` not `DataLoaderDict`.
        """
        if self.templates is not None:
            print("Templates already attached. Old templates being replaced.")

        self.templates = self(templates)
        self.label_names = names

    def predict(self, x: Samples) -> jax.Array:
        """Calculate similarity of samples `x` to templates `template`.

        Parameters
        ----------
        x : ArrayLike
            Either a row vector of a single sample or a matrix where each row
            is a sample.

        Returns
        -------
        similarity : jax.Array
            A n_samples x n_templates matrix with a similarity score between 0
            and 1 for each sample, template pair.

        See Also
        --------
        `model.label_names`

        """
        if self.templates is None:
            raise ValueError(
                """Must attach templates to model or explicitly pass templates.
                See `model.attach_templates`."""
            )

        return self.logits_fn(self(x), self.templates)

    @nnx.jit
    def logits_fn(self, x, templates):
        return x @ templates.T


class MultiLabelAccuracy(nnx.metrics.Average):
    """Accuracy metric for multilabel classification."""

    def update(self, *, predictions: jax.Array, labels: jax.Array, **_) -> None:  # type: ignore[override]
        """In-place update this ``Metric``.

        Predictions and arguments should have shape n_samples x n_labels.

        Args:
          predictions: The predictions of the model. Should be a vector of
          bools.
          labels: Ground truth labels.

        """
        if not all(
            psz == lsz for psz, lsz in zip(predictions.shape, labels.shape)
        ):
            raise ValueError(
                "Expected predictions and labels to be the same shape."
            )

        super().update(values=(predictions == labels).mean(axis=1))


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

        log_dir = "./tmp"
        # TODO: Maybe add model name here.
        self.writers = {
            k: tensorboard.SummaryWriter(os.path.join(log_dir, k))
            for k in ["train", "test"]
        }

    # @nnx.jit
    def loss_fn(
        self, model: Model, x: Samples, templates: Samples, labels: Labels
    ) -> tuple[float, jax.Array]:
        """Score the model's prediction success against known labels."""
        x = model(x)
        templates = model(templates)

        logits = model.logits_fn(x, templates)
        loss = optax.losses.sigmoid_binary_cross_entropy(logits, labels).mean()

        return loss, nnx.sigmoid(logits) > 0.5

    # @nnx.jit
    def train_step(self, batch: Batch) -> None:
        @nnx.jit
        def loss_fn(model, x, templates, y):
            x = model(x)
            templates = model(templates)

            logits = model.logits_fn(x, templates)
            loss = optax.losses.sigmoid_binary_cross_entropy(logits, y).mean()

            return loss, nnx.sigmoid(logits) > 0.5

        templates, x = self.data.split_batch(batch[0])
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, preds), grads = grad_fn(self.model, x, templates, batch[-1])
        self.metrics.update(loss=loss, predictions=preds, labels=batch[-1])
        self.optimizer.update(grads)

    # @nnx.jit
    def eval_step(self, batch: Batch) -> None:
        templates, x = self.data.split_batch(batch[0])
        loss, preds = self.loss_fn(self.model, x, templates, batch[-1])
        self.metrics.update(loss=loss, predictions=preds, labels=batch[-1])

    def write_metrics(self, stage: str, step: int):
        for metric, value in self.metrics.compute().items():
            self.history[f"{stage}_{metric}"].append(value)
            self.writers[stage].scalar(metric, value, step)

        self.metrics.reset()
        self.writers[stage].flush()

    def train(
        self,
        max_epochs: int = 1000,
        stop_delta: float = 1e-5,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train weights for the model."""

        def test_epoch():
            for batch in self.data["test"].batch():
                self.eval_step(batch)

            self.write_metrics("test", epoch)

        def train_epoch():
            for batch in self.data["train"].batch():
                self.train_step(batch)

            self.write_metrics("train", epoch)

        window_size = 20
        delta = np.full(window_size, np.nan)
        delta[0] = stop_delta + 1
        last_err = 0.0
        for epoch in range(max_epochs):
            if abs(np.nanmean(delta)) < stop_delta:
                break

            try:
                train_epoch()
                test_epoch()
            except KeyboardInterrupt:
                # End training gracefully on keyboard interrupt. Model will
                # still be trained.
                print("\nExiting training loop")
                break

            err = self.history["test_loss"][-1]
            if epoch > 0:
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

            epoch += 1

        return self.history

    def validate(self):
        pass


class RawSimilarity(Model):
    """Model without weights.

    Prediction is the dot product between the samples and templates.
    """

    def __call__(self, x):
        return x


class SingleLayer(Model):
    def __init__(self, name: str, seed: int, dims_in: int, dims_out: int):
        super().__init__(name)
        rngs = nnx.Rngs(seed)
        self.linear = nnx.Linear(dims_in, dims_out, rngs=rngs)

    def __call__(self, x: Samples):
        return nnx.gelu(self.linear(x))


class MultiLayer(Model, nnx.Module):
    def __init__(self, name: str, dims: tuple[int, ...], seed: int):
        """Multi-layer perceptron for predicting labels.

        Note: final step is dot product between samples and templates so the
        number of dimensions of the last layer is not the number of dimensions
        of the output. The true output dimensionality is determine by the
        number of rows in templates.
        """
        super().__init__(name)
        rngs = nnx.Rngs(seed)
        self.layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.gelu(layer(x))

        return x


class MLPExtras(Model):
    pass


class Attention(Model):
    pass
