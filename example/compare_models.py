import os

import jax
import optax

import abstract2gene as a2g

DATASET = "bioc_small"

MAX_GENE_TESTS = 100

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
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".25",
    }
)

data, _ = a2g.dataset.load_dataset(
    DATASET,
    seed=42,
    batch_size=124,
    labels_per_batch=4,
    template_size=16,
)

tx = optax.adam(learning_rate=1e-4)
model: a2g.model.Model = a2g.model.RawSimilarity(name="noweights")

# Coarse dimension search
dims_in = data.n_features
for d in range(1, 20, 2):
    data.reset_rngs()
    model_name = f"random_weights_{d}_dims"
    model = a2g.model.SingleLayer(
        name=model_name, seed=0, dims_in=dims_in, dims_out=d
    )
    trainer = a2g.model.Trainer(model, data, tx)
    results = trainer.train()
    trainer.test()

data, _ = a2g.dataset.load_dataset(
    DATASET,
    seed=42,
    batch_size=256,
    labels_per_batch=8,
    template_size=16,
)
dims = (data.n_features, 256, 256)
model = a2g.model.MultiLayer(name="multi", seed=20, dims=dims)
tx = optax.adam(1e-4)
trainer = a2g.model.Trainer(model, data, tx)
results = trainer.train()
df = trainer.test()
trainer.plot(df)
