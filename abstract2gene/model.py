"""Models for comparing abstract features.

Various models for comparing abstract embeddings. This generally means
comparing an individual publication's abstract embeddings to template
embeddings for different labels (in the case of this package, genes).

Templates are the average of many examples of abstracts tagged with a label.
"""

__ALL__ = ["ModelNoWeights", "ModelMSELoss", "ModelMaximizeDistance"]

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import tree_util
from sklearn.model_selection import LeaveOneOut

from .dataset import DataSet
from .typing import ArrayLike, Batch, LabelLike, PyTree

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

    Train the model by calling the `train` method.

    Use the model by calling the `predict` method (after training).
    """

    def __init__(
        self,
        name: str = "",
        n_dims: int = 20,
    ):
        """Initialize a model.

        Parameters
        ----------
        features, labels : numpy.ndarray
            The data to train on. Both should have the same number of rows
            (independent samples).
        symbols : numpy.ndarray
            An object type array of label names.
        seed : int
            Seed for the model's random number generator. Run `reset_rng` to
            reseed the model with the value used to create the model.
        train_test_val : tuple[float, float, float], default (0.8, 0.1, 0.1)
            Proportional of labels to use for training, testing, and
            validation.
        name : str, optional
            A name used create a results filename.
        batch_size : int, default 64
            How many examples to train on for each label. Half the examples
            will be labeled with the current label and the other half will be
            randomly selected for the pool of samples not associated with the
            current label.
        n_validation_labels : int, default 10
            Number of labels to be used for validation instead of training.
        n_template_samples : int, default 32
            How many examples to use when creating the templates for a given
            label. A label's template is created by averaging positive examples
            of features associated with the given label.

        Note: There must be at least `(batch_size // 2) + n_template_samples`
        samples tagged with each label. So labels should be filtered beforehand
        to only those with enough examples.

        """
        self.name = name
        self.result_file = RESULT_TEMPLATE.format(name=name)
        self.templates: ArrayLike | None = None
        self.params: PyTree = {}
        self.n_dims = n_dims

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.predict(x)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        self.result_file = RESULT_TEMPLATE.format(name=value)

    def predict(
        self, x: ArrayLike, templates: ArrayLike | None = None
    ) -> ArrayLike:
        """Calculate similarity of samples `x` to templates `template`.

        Parameters
        ----------
        x : ArrayLike (numpy.ndarray, jax.ndarray)
            Either a row vector of a single sample or a matrix where each row
            is a sample.
        templates : ArrayLike (numpy.ndarray, jax.ndarray)
            Either a row vector of a single template or a matrix where each row
            is a separate template.

        Returns
        -------
        similarity : ArrayLike (matching inputs)
            A n_samples x n_templates matrix with a similarity score between 0
            and 1 for each sample, template pair.

        """
        templates = templates if templates is not None else self.templates
        if templates is None:
            raise ValueError(
                "Must install templates to model or explicitly pass templates."
            )

        return self._predict(x, templates)

    def _predict(self, x: ArrayLike, templates: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def loss(
        self, x: ArrayLike, templates: ArrayLike, labels: LabelLike
    ) -> float:
        """Score the model's prediction success against known labels."""
        raise NotImplementedError

    def gradient(
        self, x: ArrayLike, templates: ArrayLike, labels: LabelLike
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

    def train(
        self,
        data: DataSet,
        max_epochs: int = 1000,
        stop_delta: float = 1e-5,
        learning_rate: float = 0.002,
        reset_weights: bool = True,
        verbose: bool = True,
    ):
        """Train weights for the model."""

        def validate() -> float:
            n = data.n_validate
            return sum(self.loss(*batch) for batch in data.validate()) / n

        if (not self.params) or reset_weights:
            self.params = self.init_weights(data, self.n_dims, learning_rate)

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
            pred_t = 0
            pred_f = 0
            pred_tsd = 0
            pred_fsd = 0
            try:
                for batch in data.train():

                    if count < window_size:
                        x, templates, labels = batch
                        x_t = x[labels.squeeze(), :]
                        x_f = x[np.logical_not(labels.squeeze()), :]
                        prediction = self.predict(x_t, templates)
                        pred_t += prediction.mean()
                        pred_tsd += prediction.std()
                        prediction = self.predict(x_f, templates)
                        pred_f += prediction.mean()
                        pred_fsd += prediction.std()

                    self.update(batch, learning_rate)
                    train_err += self.loss(*batch)
                    count += 1
            except KeyboardInterrupt:
                # End training gracefully on keyboard interrupt. Model will
                # still be trained.
                print("\nExiting training loop")
                break

            # learning_rate *= e ** (-learning_decay)
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
                    + f"{pred_tsd / window_size:.4g}"
                )
                print(
                    f"  False: {pred_f / window_size:.4g} +/- "
                    + f"{pred_fsd / window_size:.4g}"
                )
                print(
                    "  Distance: ",
                    f"{(pred_t - pred_f) / (0.5 * (pred_tsd + pred_fsd)):.4g}",
                )

            last_err = err
            epoch += 1

    def init_weights(
        self, data: DataSet, n_dims: int, learning_rate: float
    ) -> PyTree:
        params = {
            "w": data._rng.normal(0, 0.01, (data.features.shape[1], n_dims))
        }
        return params["w"] / np.linalg.norm(params["w"], axis=0, keepdims=True)

    def test(
        self,
        data: DataSet,
        max_num_tests: int | None = None,
        save_results: bool = True,
    ):
        """Tests how well the model differentiates labels.

        For each label, compares predictions for `self.batch_size` labeled
        samples against samples without the label. Templates are created using
        a leave-one-out method. Results are stored as a table in
        `self.result_file`.

        The `max_num_tests` controls how many labels to test. If None, tests
        are test labels.

        """
        if save_results and os.path.exists(self.result_file):
            os.unlink(self.result_file)

        labels_test = data.labels[:, data._masks["test"]]
        if data.label_names is not None:
            symbols_test = data.label_names[data._masks["test"]]
        else:
            symbols_test = None

        max_num_tests = max_num_tests or labels_test.shape[1]
        n_tests = min(max_num_tests, labels_test.shape[1])

        loss = 0.0
        for i in range(n_tests):
            symbol = symbols_test[i] if symbols_test is not None else "Unknown"
            loss += self._test_label(
                data, labels_test[:, i], symbol, save_results
            )

        if save_results and not os.path.exists(RESULTS_TABLE):
            with open(RESULTS_TABLE, "w") as f:
                f.write("name\tmean_distance\n")

        if save_results:
            with open(RESULTS_TABLE, "a") as f:
                f.write(f"{self.name}\t{loss / n_tests:0.4g}\n")

        print(f"Average sample mean distance:\n  {loss / n_tests:0.4g}")

    def _test_label(self, data, indices, symbol, save_results):
        features_label = data.features[indices, :]
        # Make all tests have same number of samples.
        features_label = features_label[: data.batch_size, :]
        features_other = data.features[np.logical_not(indices), :]

        if data.feature_names is not None:
            label_pmids = data.feature_names[indices]
            unlabeled_pmids = data.feature_names[np.logical_not(indices)]
            pmids = np.concat(
                (
                    label_pmids[: data.batch_size],
                    unlabeled_pmids[: data.batch_size],
                )
            )
        else:
            pmids = np.repeat("Unknown", data.batch_size * 2)

        loo = LeaveOneOut()
        sim_within = np.zeros((data.batch_size))
        sim_between = np.zeros((data.batch_size))
        for i, (train_index, test_index) in enumerate(
            loo.split(features_label)
        ):
            template = features_label[train_index, :].mean(
                axis=0, keepdims=True
            )
            sim_within[i] = self.predict(
                features_label[test_index, :], template
            )[0, 0]
            sim_between[i] = self.predict(features_other[[i], :], template)[
                0, 0
            ]

        if save_results:
            label_df = pd.DataFrame(
                {
                    "label": np.repeat(symbol, data.batch_size * 2),
                    "pmid": pmids,
                    "group": np.concat(
                        (
                            np.asarray("within").repeat(data.batch_size),
                            np.asarray("between").repeat(data.batch_size),
                        )
                    ),
                    "similarity": np.concat((sim_within, sim_between)),
                }
            )

            header = not os.path.exists(self.result_file)
            label_df.to_csv(
                self.result_file,
                sep="\t",
                index=False,
                mode="a",
                header=header,
            )

        distance = sim_within.mean() - sim_between.mean()
        stderr = np.concat((sim_within, sim_between)).std(ddof=1)
        stderr /= data.batch_size * 2
        return distance / stderr


class ModelNoWeights(Model):
    """Model without weights.

    Prediction is the dot product between the samples and templates.
    """

    def train(self, *args, **kwds):
        pass

    def _predict(self, x: ArrayLike, templates: ArrayLike) -> ArrayLike:
        return x @ templates.T


class ModelMSELoss(Model):
    """Model based on the mean squared error of y_hat - y loss function."""

    def gradient(
        self, x: ArrayLike, templates: ArrayLike, labels: LabelLike
    ) -> PyTree:
        """Calculate gradient of loss function with respect to weights.

        Note: labels are assumed to be a vector. When using the model itself
        labels will likely be a matrix of samples x n_genes. When training only
        one gene is tested at a given time. Due to the math, `x`, `template`,
        and `labels` must all be vectors while `weights` is a matrix. It is
        expected that `x` is actually a samples x features matrix and the
        matrix operations are iterated over it's samples.
        """
        return {
            "w": sum(
                (self.predict(x.reshape((1, -1)), templates) - label)
                * (
                    x.reshape((-1, 1)) @ ([templates] @ self.params["w"])
                    + templates.T @ (x.reshape((1, -1)) @ self.params["w"])
                )
                for x, label in zip(x, labels)
            )
            / x.shape[0]
        }

    def _predict(self, x: ArrayLike, templates: ArrayLike) -> ArrayLike:
        return x @ self.params["w"] @ self.params["w"].T @ templates.T

    def loss(
        self, x: ArrayLike, templates: ArrayLike, labels: LabelLike
    ) -> float:
        return np.square(self.predict(x, templates) - labels).mean()


@jax.jit
def _flax_predict(
    weights: ArrayLike, samples: ArrayLike, templates: ArrayLike
) -> ArrayLike:
    return (samples @ weights) @ (weights.T @ templates.T)


@jax.jit
def _flax_loss(
    params: PyTree,
    samples: ArrayLike,
    templates: ArrayLike,
    labels: LabelLike,
):
    prediction = _flax_predict(params["w"], samples, templates)
    return jnp.mean(optax.losses.l2_loss(prediction, labels))


_flax_gradient = jax.grad(_flax_loss, has_aux=False)


class ModelJax(Model):
    def __init__(self, *args, seed: int = 0, **kwds):
        super().__init__(*args, **kwds)
        self._key = jax.random.PRNGKey(seed)

    def _predict(self, samples, templates):
        return _flax_predict(self.params["w"], samples, templates)

    def loss(self, samples, templates, labels):
        return _flax_loss(self.params, samples, templates, labels)

    def init_weights(
        self, data: DataSet, n_dims: int, learning_rate: float
    ) -> PyTree:
        self._optimizer = optax.adam(learning_rate=learning_rate)
        self._key, key = jax.random.split(self._key)
        params = {"w": jax.random.normal(key, (data.n_features, n_dims))}
        self._state = self._optimizer.init(params)

        return params

    def gradient(
        self, x: ArrayLike, templates: ArrayLike, labels: LabelLike
    ) -> PyTree:
        return _flax_gradient(self.params, x, templates, labels)

    def update(self, batch: Batch, learning_rate: float | None) -> None:
        grads = self.gradient(*batch)
        updates, self._state = self._optimizer.update(grads, self._state)
        self.params = optax.apply_updates(self.params, updates)
