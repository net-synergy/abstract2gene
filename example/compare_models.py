import os

import datasets
import jax
import numpy as np
import optax
from sentence_transformers import SentenceTransformer

import abstract2gene as a2g
from abstract2gene.data import dataset_path, model_path

# from abstract2gene.dataset import mock_dataloader

DATASET = "bioc_small"

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_enable_triton_gemm=true "
    "--xla_gpu_graph_level=0 "
)

os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".4",
    }
)

encoder = SentenceTransformer(model_path("specter-abstract-genes"))

dataset = datasets.load_from_disk(dataset_path(DATASET))
genes = np.bincount(jax.tree.leaves(dataset["gene"]))
mask = genes > np.quantile(genes, 0.75)
gene_ids = np.arange(len(genes))[mask]

dataset = dataset.filter(
    lambda example: any(np.isin(example["gene"], gene_ids)),
    num_proc=10,
).map(
    lambda example: {"embedding": encoder.encode(example["abstract"])},
    remove_columns=["abstract"],
)

data, _ = a2g.dataset.from_huggingface(
    dataset,
    seed=42,
    batch_size=128,
    labels_per_batch=4,
    template_size=16,
    max_steps=2000,
)

data.update_params(batch_size=128, labels_per_batch=4, template_size=1)
# data = mock_dataloader(n_classes=100, n_features=200, n_samples=10000, noise=0)

# tx = optax.sgd(learning_rate=3e-4)
model: a2g.model.Model = a2g.model.RawSimilarity(name="noweights", seed=0)

dims = (data.n_features, 768, 768)
model = a2g.model.MLPExtras(name="single", seed=20, dims=dims)
data.reset_rngs()
tx = optax.adam(learning_rate=1e-4)
trainer = a2g.model.Trainer(model, data, tx)
results = trainer.train(max_epochs=30)

## TEMP: Will fix in next commit.
# data.update_params(template_size=32)
# df = trainer.test()
# trainer.plot(df, "multi_layer_large_templates.svg")
