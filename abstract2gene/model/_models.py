__all__ = [
    "RawSimilarity",
    "SingleLayer",
    "MultiLayer",
    "MLPExtras",
    "Attention",
    "Model",
]

import jax
from flax import nnx

from abstract2gene.typing import Names, Samples

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

        return nnx.sigmoid(self.logits_fn(self(x), self.templates))

    @nnx.jit
    def logits_fn(self, x, templates):
        return x @ templates.T


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

    @nnx.jit
    def __call__(self, x):
        for layer in self.layers:
            x = nnx.gelu(layer(x))

        return x


class MLPExtras(Model):
    pass


class Attention(Model):
    pass
