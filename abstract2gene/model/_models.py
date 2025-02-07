__all__ = [
    "RawSimilarity",
    "MultiLayer",
    "MLPExtras",
    "Attention",
    "Model",
]

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike

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

    def predict(self, x: Samples, templates: Samples | None) -> jax.Array:
        """Calculate similarity of samples `x` to templates `template`.

        Parameters
        ----------
        x : ArrayLike
            Either a row vector of a single sample or a matrix where each row
            is a sample.
        templates : ArrayLike, optional
            A set of templates to compare x to. If None, templates fixed to the
            model will be used.

        Returns
        -------
        similarity : jax.Array
            A n_samples x n_templates matrix with a similarity score between 0
            and 1 for each sample, template pair.

        See Also
        --------
        `model.label_names`

        """
        templates = templates if templates is not None else self.templates

        if templates is None:
            raise ValueError(
                """Must attach templates to model or explicitly pass templates.
                See `model.attach_templates`."""
            )

        return nnx.sigmoid(self.logits_fn(self(x), templates))

    @nnx.jit
    def logits_fn(self, x: Samples, templates: Samples):
        return x @ templates.T


class LinearTemplate(nnx.Module):
    """A linear transformation between inputs and predefined weights.

    For use with predefined templates, which act as the weights of the linear
    layer. Output is `x @ templates + bias` where bias is a trainable scalar
    applied to all output dimensions.
    """

    def __init__(self, rngs: rnglib.Rngs, dtype: DTypeLike = jnp.float32):
        bias_key = rngs.params()
        initializer = nnx.initializers.zeros_init()
        self.bias = nnx.Param(initializer(bias_key, (1,), dtype))
        self.templates: Samples | None = None

    def fix(self, templates: Samples):
        self.templates = templates

    def __call__(self, x: Samples, templates: Samples | None):
        templates = templates if templates is not None else self.templates

        if templates is None:
            raise ValueError(
                "No templates attached to this layer. Either pass templates"
                + "in or attach with the fix method."
            )

        return (x @ templates.T) + self.bias.value


class RawSimilarity(Model):
    """Model without weights.

    Prediction is the dot product between the samples and templates.
    """

    def __init__(self, name: str, seed: int):
        super().__init__(name)
        rngs = nnx.Rngs(seed)
        self.template = LinearTemplate(rngs=rngs)

    def __call__(self, x: Samples):
        return x

    def logits_fn(self, x: Samples, templates: Samples):
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        templates = templates / jnp.linalg.norm(
            templates, axis=1, keepdims=True
        )
        return self.template(x, templates)


class MultiLayer(Model):
    def __init__(self, name: str, dims: tuple[int, ...], seed: int):
        """Multi-layer perceptron for predicting labels.

        Adds dense layers after the embedding.

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
        self.template = LinearTemplate(rngs=rngs)

    @nnx.jit
    def __call__(self, x):
        for layer in self.layers:
            x = nnx.gelu(layer(x))

        return x

    def logits_fn(self, x: Samples, templates: Samples):
        return self.template(x, templates)


class MLPExtras(Model):
    def __init__(self, name: str, dims: tuple[int, ...], seed: int):
        """Multi-layer perceptron for predicting labels.

        Like MultiLayer but adds batch normalization to dense layers. This is
        showing improvement in the average prediction for each gene. Without
        this each genes predictions have a gene specific mean in variance. The
        batch normalization is reducing this to make predictions on a constant
        threshold value more viable.

        Note: final step is dot product between samples and templates so the
        number of dimensions of the last layer is not the number of dimensions
        of the output. The true output dimensionality is determine by the
        number of rows in templates.
        """
        super().__init__(name)
        rngs = nnx.Rngs(seed)
        self.template = LinearTemplate(rngs=rngs)
        self.layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ]
        self.normal = [
            nnx.BatchNorm(dims[i], rngs=rngs) for i in range(1, len(dims))
        ]

    @nnx.jit
    def __call__(self, x):
        for norm, layer in zip(self.normal, self.layers):
            x = nnx.gelu(norm(layer(x)))

        return x

    def logits_fn(self, x: Samples, templates: Samples):
        return self.template(x, templates)


class Attention(Model):
    pass
