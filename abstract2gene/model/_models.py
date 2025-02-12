__all__ = [
    "RawSimilarity",
    "MultiLayer",
    "MLPExtras",
    "Model",
    "load_from_disk",
]


import dataclasses
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from sentence_transformers import SentenceTransformer

from abstract2gene.data import model_path
from abstract2gene.dataset import DataLoaderDict
from abstract2gene.typing import Samples


@dataclasses.dataclass
class Templates:
    """Holds template data.

    Only needed since directly storing arrays in an nnx model that aren't
    parameters causes an error when jax.flatten is run.
    """

    indices: np.ndarray
    values: jax.Array


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
        seed: int = 0,
    ):
        """Initialize a model.

        Parameters
        ----------
        encoder : SentenceTransformer, Optional
            If set, this will be used to produce embeddings when strings are
            passed in.

        """
        self.templates: Templates | None = None
        self.encoder: SentenceTransformer | None = None
        self.seed = seed

        self._encoder_name = ""

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def attach_templates(
        self, dataset: DataLoaderDict, template_size: int | None = None
    ) -> None:
        """Add a dataset's templates to the model.

        These are used for prediction---outside of training. During training,
        templates are created with batches by the dataset.
        """
        if self.templates is not None:
            print("Templates already attached. Old templates being replaced.")

        templates, indices = dataset.get_templates(template_size)
        templates = self(templates)
        templates = dataset.fold_templates(templates, template_size).mean(
            axis=1
        )

        self.templates = Templates(indices=indices, values=templates)

    def attach_encoder(self, name: str) -> None:
        """Attach a sentence-transformer encoder.

        Pass in the name of the encoder exactly as would be given to the
        SentenceTransformer class.
        """
        self._encoder_name = name
        self.encoder = SentenceTransformer(name)

    def predict(
        self,
        x: Samples | str | list[str],
        templates: Samples | None = None,
    ) -> jax.Array:
        """Calculate similarity of samples `x` to templates `template`.

        Parameters
        ----------
        x : ArrayLike, str, list[str]
            Can be embeddings in the form of a row vector of a single sample or
            a matrix where each row is a sample. Or can be strings to be passed
            to the encoder to create embeddings (requires encoder to be
            attached).
        templates : ArrayLike, optional
            A set of templates to compare x to. If None, templates fixed to the
            model will be used.

        Returns
        -------
        similarity : jax.Array
            A n_samples x n_templates matrix with a similarity score between 0
            and 1 for each sample, template pair.

        """
        if templates is None and self.templates is None:
            raise ValueError(
                """Must attach templates to model or explicitly pass templates.
                See `model.attach_templates`."""
            )

        templates = (
            templates
            if templates is not None
            else self.templates.values  # ignore[union-attr]
        )

        if isinstance(x, str):
            x = [x]

        if not isinstance(x, jax.Array):
            if self.encoder is None:
                raise ValueError(
                    """Must attach an encoder to predict from strings."""
                )

            x = jnp.asarray(self.encoder.encode(x))

        return nnx.sigmoid(
            self.logits_fn(self(x), templates)  # ignore[arg-type]
        )

    @nnx.jit
    def logits_fn(self, x: Samples, templates: Samples) -> jax.Array:
        return x @ templates.T

    def save_to_disk(self, name: str) -> None:
        save_dir = ocp.test_utils.erase_and_create_empty(model_path(name))

        cls_name = str(type(self))
        m = re.search(r"_models\.(.*)'", cls_name)

        if m is None:
            raise RuntimeError("Could not determine model class name.")

        metadata = {"seed": self.seed, "cls_name": m.group(1)}
        if hasattr(self, "_dims"):
            metadata["dims"] = self._dims

        metadata["encoder"] = self._encoder_name

        with open(os.path.join(save_dir, "metadata.json"), "w") as js:
            json.dump(metadata, js)

        if self.templates is not None:
            np.savez_compressed(
                os.path.join(save_dir, "templates.npz"),
                templates=np.asarray(self.templates.values),
                label_indices=self.templates.indices,
            )

        _, state = nnx.split(self)

        with ocp.StandardCheckpointer() as ckptr:
            ckptr.save(os.path.join(save_dir, "state"), state)
            ckptr.wait_until_finished()


def load_from_disk(name: str) -> Model:
    save_dir = model_path(name)

    with open(os.path.join(save_dir, "metadata.json"), "r") as js:
        metadata = json.load(js)

    model_cls = {
        "RawSimilarity": RawSimilarity,
        "MultiLayer": MultiLayer,
        "MLPExtras": MLPExtras,
    }

    cls_name = metadata.pop("cls_name")
    encoder_name = metadata.pop("encoder")
    print(metadata)

    abstract_model = nnx.eval_shape(lambda: model_cls[cls_name](**metadata))
    graphdef, abstract_state = nnx.split(abstract_model)
    with ocp.StandardCheckpointer() as ckptr:
        state = ckptr.restore(os.path.join(save_dir, "state"), abstract_state)

    model = nnx.merge(graphdef, state)

    if encoder_name:
        model.encoder = SentenceTransformer(encoder_name)
        model._encoder_name = encoder_name

    if os.path.exists(os.path.join(save_dir, "templates.npz")):
        with np.load(os.path.join(save_dir, "templates.npz")) as data:
            templates = data["templates"]
            label_indices = data["label_indices"]

        model.templates = Templates(indices=label_indices, values=templates)

    return model


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

    def __call__(self, x: Samples, templates: Samples):
        return (x @ templates.T) + self.bias.value


class RawSimilarity(Model):
    """Model without weights.

    Prediction is the dot product between the samples and templates.
    """

    def __init__(self, seed: int):
        super().__init__(seed=seed)
        rngs = nnx.Rngs(seed)
        self.template_layer = LinearTemplate(rngs=rngs)

    def __call__(self, x: Samples):
        return x

    def logits_fn(self, x: Samples, templates: Samples):
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        templates = templates / jnp.linalg.norm(
            templates, axis=1, keepdims=True
        )
        return self.template_layer(x, templates)


class MultiLayer(Model):
    def __init__(self, dims: tuple[int, ...], seed: int):
        """Multi-layer perceptron for predicting labels.

        Adds dense layers after the embedding.

        Note: final step is dot product between samples and templates so the
        number of dimensions of the last layer is not the number of dimensions
        of the output. The true output dimensionality is determine by the
        number of rows in templates.
        """
        super().__init__(seed=seed)
        rngs = nnx.Rngs(seed)
        self._dims = dims

        self.layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ]
        self.template_layer = LinearTemplate(rngs=rngs)

    @nnx.jit
    def __call__(self, x):
        for layer in self.layers:
            x = nnx.gelu(layer(x))

        return x

    def logits_fn(self, x: Samples, templates: Samples):
        return self.template_layer(x, templates)


class MLPExtras(Model):
    def __init__(self, dims: tuple[int, ...], seed: int):
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
        super().__init__(seed=seed)
        rngs = nnx.Rngs(seed)
        self._dims = dims
        self.template_layer = LinearTemplate(rngs=rngs)
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
        return self.template_layer(x, templates)
