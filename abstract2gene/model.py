"""Models for comparing abstract features.

Various models for comparing abstract embeddings. This generally means
comparing an individual publication's abstract embeddings to template
embeddings for different labels (in the case of this package, genes).

Templates are the average of many examples of abstracts tagged with a label.
"""

__all__ = [
    "ModelNoWeights",
    "ModelSingleLayer",
    "ModelMultiLayer",
    "train",
    "test",
]

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx
from jax import tree_util

from .dataset import DataLoader
from .typing import Batch, Features, Labels, Names, PyTree

RESULT_TEMPLATE = "results/{name}_validation.tsv"
RESULTS_TABLE = "results/model_comparison.tsv"


class Model:
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
        self.templates: Features | None = None
        self.label_names: Names | None = None
        self.params: PyTree = {}

    def __call__(self, x: Features) -> jax.Array:
        return self.predict(x)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        self.result_file = RESULT_TEMPLATE.format(name=value)

    def attach_templates(self, dataset: DataLoader) -> None:
        """Add the templates for the model.

        These are used for prediction---outside of training. During training,
        templates are created with batches by the dataset.
        """
        self.templates, self.label_names = dataset.get_templates()

    def predict(self, x: Features) -> jax.Array:
        """Calculate similarity of samples `x` to templates `template`.

        Parameters
        ----------
        x : ArrayLike (numpy.ndarray, jax.ndarray)
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

        return self._predict(x, self.templates)

    def _predict(self, x: Features, templates: Features) -> jax.Array:
        """Variant of predict to use during training."""
        raise NotImplementedError

    def loss(self, x: Features, templates: Features, labels: Labels) -> float:
        """Score the model's prediction success against known labels."""
        raise NotImplementedError

    def gradient(
        self, x: Features, templates: Features, labels: Labels
    ) -> PyTree:
        """Calculate the loss function's gradient with respect to weights."""
        raise NotImplementedError

    def update(self, batch: Batch, learning_rate: float) -> None:
        """Update the model's weights based on the loss gradient."""
        tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            self.params,
            self.gradient(*batch),
        )

    def init_weights(self, data: DataLoader, learning_rate: float) -> None:
        raise NotImplementedError


def train(
    model: Model,
    data: DataLoader,
    max_epochs: int = 1000,
    stop_delta: float = 1e-5,
    learning_rate: float = 0.002,
    reset_weights: bool = True,
    verbose: bool = True,
):
    """Train weights for the model."""

    def validate() -> float:
        n = data.n_validate
        return sum(model.loss(*batch) for batch in data.validate()) / n

    if (not model.params) or reset_weights:
        model.init_weights(data, learning_rate)

    epoch = 0
    window_size = 20
    delta = np.full(window_size, np.nan)
    last_err = validate()
    if verbose:
        print(f"Initial validation loss: {last_err:0.4g}")

    delta[0] = stop_delta + 1
    while (epoch < max_epochs) and (abs(np.nanmean(delta)) > stop_delta):
        train_err = 0.0
        count = 0
        pred_t = 0.0
        pred_f = 0.0
        pred_tvar = 0.0
        pred_fvar = 0.0
        try:
            for batch in data.train():
                if count < window_size:
                    x, templates, labels = batch
                    col = np.argmax(labels.sum(axis=0))
                    x_t = x[labels[:, col].squeeze(), :]
                    x_f = x[np.logical_not(labels[:, col].squeeze()), :]
                    prediction = model._predict(x_t, templates)
                    pred_t += prediction.mean().item()
                    pred_tvar += prediction.var().item()
                    prediction = model._predict(x_f, templates)
                    pred_f += prediction.mean().item()
                    pred_fvar += prediction.var().item()

                model.update(batch, learning_rate)
                train_err += model.loss(*batch)
                count += 1
        except KeyboardInterrupt:
            # End training gracefully on keyboard interrupt. Model will
            # still be trained.
            print("\nExiting training loop")
            break

        err = validate()
        delta[epoch % window_size] = last_err - err

        if verbose:
            print(f"Epoch: {epoch}")
            print(f"  Training loss: {train_err / count:.4g}")
            print(f"  Validation loss: {err:.4g}")
            print(f"  Delta: {delta[epoch % window_size]:.4g}")
            print(f"  Avg delta: {delta.mean():.4g}")
            print(
                f"  True: {pred_t / window_size:.4g} +/- "
                + f"{np.sqrt(pred_tvar / window_size):.4g}"
            )
            print(
                f"  False: {pred_f / window_size:.4g} +/- "
                + f"{np.sqrt(pred_fvar / window_size):.4g}"
            )
            distance = (pred_t - pred_f) / np.sqrt((pred_tvar + pred_fvar))
            print(f"  Distance: {distance:.4g}")

        last_err = err
        epoch += 1


def test(
    model: Model,
    data: DataLoader,
    max_num_tests: int | None = None,
    save_results: bool = True,
):
    """Tests how well the model differentiates labels.

    For each label, compares predictions for `model.batch_size` labeled
    samples against samples without the label. Templates are created using
    a leave-one-out method. Results are stored as a table in
    `model.result_file`.

    The `max_num_tests` controls how many labels to test. If None, tests
    are test labels.

    """

    def _test_label(model, batch, symbol, pmids, save_results):
        labels = batch[-1]
        y_hat = model._predict(*batch[:-1])
        sim_within = y_hat[labels]
        sim_between = y_hat[np.logical_not(labels)]

        if save_results:
            label_df = pd.DataFrame(
                {
                    "label": np.repeat(symbol, labels.shape[0]),
                    "pmid": pmids,
                    "group": np.concat(
                        (
                            np.asarray("within").repeat(labels.sum()),
                            np.asarray("between").repeat(
                                np.logical_not(labels).sum()
                            ),
                        )
                    ),
                    "similarity": np.concat((sim_within, sim_between)),
                }
            )

            header = not os.path.exists(model.result_file)
            label_df.to_csv(
                model.result_file,
                sep="\t",
                index=False,
                mode="a",
                header=header,
            )

        distance = sim_within.mean() - sim_between.mean()
        stderr = np.concat((sim_within, sim_between)).std(ddof=1)
        stderr /= np.sqrt(labels.shape[0] // 2)
        return distance / stderr

    if save_results and os.path.exists(model.result_file):
        os.unlink(model.result_file)

    max_num_tests = max_num_tests or data.n_test
    n_tests = min(max_num_tests, data.n_test)

    loss = 0.0
    for batch in data.test():
        symbol = data.batch_label_name()
        pmids = data.batch_sample_names()
        loss += _test_label(model, batch, symbol, pmids, save_results)

    if save_results and not os.path.exists(RESULTS_TABLE):
        with open(RESULTS_TABLE, "w") as f:
            f.write("name\tmean_distance\n")

    if save_results:
        with open(RESULTS_TABLE, "a") as f:
            f.write(f"{model.name}\t{loss / n_tests:0.4g}\n")

    print(f"Average sample mean distance:\n  {loss / n_tests:0.4g}")


class ModelNoWeights(Model):
    """Model without weights.

    Prediction is the dot product between the samples and templates.
    """

    def _predict(self, x: Features, templates: Features) -> jax.Array:
        return x @ templates.T

    def loss(self, samples, templates, labels):
        return 0.0

    def init_weights(self, data, learning_rate):
        pass

    def update(self, batch: Batch, learning_rate: float | None) -> None:
        pass


@jax.jit
def _sl_predict(
    weights: jax.Array, samples: Features, templates: Features
) -> jax.Array:
    return (samples @ weights) @ (weights.T @ templates.T)


@jax.jit
def _mse_loss(
    params: PyTree,
    samples: Features,
    templates: Features,
    labels: Labels,
):
    prediction = _sl_predict(params["w"], samples, templates)
    return jnp.mean(optax.losses.l2_loss(prediction, labels))


_mse_gradient = jax.grad(_mse_loss, has_aux=False)


class ModelSingleLayer(Model):
    def __init__(self, *args, seed: int = 0, n_dims: int = 20, **kwds):
        super().__init__(*args, **kwds)
        self.n_dims = n_dims
        self._key = jax.random.PRNGKey(seed)

    def _predict(self, samples, templates):
        return _sl_predict(self.params["w"], samples, templates)

    def loss(self, samples, templates, labels):
        return _mse_loss(self.params, samples, templates, labels)

    def init_weights(self, data: DataLoader, learning_rate: float) -> None:
        self._optimizer = optax.adam(learning_rate=learning_rate)
        self._key, key = jax.random.split(self._key)
        self.params = {
            "w": jax.random.normal(key, (data.n_features, self.n_dims))
        }
        self._state = self._optimizer.init(self.params)

    def update(self, batch: Batch, learning_rate: float | None) -> None:
        grads = _mse_gradient(self.params, *batch)
        updates, self._state = self._optimizer.update(grads, self._state)
        self.params = optax.apply_updates(self.params, updates)


@jax.jit
def _ml_predict(samples, templates):
    return samples @ templates.T


class ModelMultiLayer(Model, nnx.Module):
    def __init__(
        self,
        *args,
        dims: tuple[int, ...],
        seed: int = 0,
        dropout: tuple[float, float] = (0.2, 0.1),
        **kwds,
    ):
        """Multi-layer perceptron for predicting labels.

        Note: final step is dot product between samples and templates so the
        number of dimensions of the last layer is not the number of dimensions
        of the output. The true output dimensionality is determine by the
        number of rows in templates.
        """
        super().__init__(*args, **kwds)
        self._rng = nnx.Rngs(seed)
        self.layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=self._rng)
            for i in range(len(dims) - 1)
        ]
        self.dropouts = [
            nnx.Dropout(p, rngs=self._rng)
            for p in np.linspace(*dropout, num=len(dims))
        ]
        self.activation = nnx.relu

    def _net(self, x):
        for dropout, layer in zip(self.dropouts, self.layers):
            x = self.activation(dropout(layer(x)))

        return x

    def _predict(self, samples, templates):
        samples = self._net(samples)
        samples /= jnp.linalg.norm(samples, axis=1, keepdims=True)

        templates = self._net(templates)
        templates /= jnp.linalg.norm(templates, axis=1, keepdims=True)

        return _ml_predict(samples, templates)

    def loss(self, samples, templates, labels):
        return jnp.mean(
            optax.losses.l2_loss(self._predict(samples, templates), labels)
        )

    def init_weights(self, data: DataLoader, learning_rate: float) -> None:
        self._optimizer = nnx.Optimizer(
            self, optax.adam(learning_rate=learning_rate)
        )

    @nnx.jit
    def update(self, batch: Batch, learning_rate: float | None) -> None:
        def loss(model):
            y_pred = model._predict(*batch[:-1])
            return jnp.mean((y_pred - batch[-1]) ** 2)

        grads = nnx.grad(loss)(self)
        self._optimizer.update(grads)
