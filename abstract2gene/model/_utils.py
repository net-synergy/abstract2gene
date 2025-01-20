__all__ = ["MultiLabelAccuracy", "loss_fn", "train_step", "eval_step"]

import jax
import optax
from flax import nnx

from abstract2gene.typing import Batch, Labels, Samples

from ._models import Model


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


@nnx.jit
def loss_fn(
    model: Model, x: Samples, templates: Samples, labels: Labels
) -> tuple[float, jax.Array]:
    """Score the model's prediction success against known labels."""
    x = model(x)

    logits = model.logits_fn(x, templates)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, labels).mean()

    return loss, nnx.sigmoid(logits) > 0.5


@nnx.jit
def train_fn(
    model: Model,
    x: Samples,
    templates: Samples,
    labels: Labels,
) -> tuple[float, jax.Array, jax.Array]:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(model, x, templates, labels)

    return loss, preds, grads


@nnx.jit
def eval_fn(
    model: Model, x: Samples, templates: Samples, labels: Labels
) -> tuple[float, jax.Array]:
    return loss_fn(model, x, templates, labels)
