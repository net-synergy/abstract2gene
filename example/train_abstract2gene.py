import os

import datasets
import jax
import numpy as np
import optax
from sentence_transformers import SentenceTransformer

import abstract2gene as a2g
import example._config as cfg
from abstract2gene.data import model_path
from abstract2gene.dataset import mutators
from example._logging import log, set_log

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

EXPERIMENT = "train_abstract2gene"
seed = cfg.seeds[EXPERIMENT]
set_log(EXPERIMENT)

encoder_loc = f"{cfg.hf_user}/pubmedncl-abstract2gene"
encoder = SentenceTransformer(encoder_loc)

dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts",
    data_files=cfg.A2G_TRAIN_FILES,
)["train"]

genes = np.bincount(jax.tree.leaves(dataset["gene"]))
mask = genes > np.quantile(genes, 0.05)
gene_ids = np.arange(len(genes))[mask]

dataset = dataset.filter(
    lambda example: any(np.isin(example["gene"], gene_ids)),
    num_proc=10,
)
dataset = mutators.mask_abstract(dataset, "gene").map(
    lambda example: {
        "embedding": encoder.encode(
            example["title"] + "[SEP]" + example["abstract"]
        )
    },
    remove_columns=["abstract"],
)

dataloader, _ = a2g.dataset.from_huggingface(
    dataset,
    seed=seed,
    labels="gene",
    batch_size=128,
    labels_per_batch=8,
    template_size=1,
    max_steps=3000,
)

dims = (dataloader.n_features, 768)
for n in range(1, 7):
    dataloader.reset_rngs()
    model = a2g.model.MultiLayer(seed=seed + 1, dims=dims)
    tx = optax.adam(learning_rate=1e-4)
    trainer = a2g.model.Trainer(model, dataloader, tx)

    trainer.data.update_params(labels_per_batch=2**n, template_size=1)
    results = trainer.train(max_epochs=40)

    trainer.data.update_params(template_size=4)
    results = trainer.train(max_epochs=20)

    trainer.data.update_params(template_size=8)
    results = trainer.train(max_epochs=20)

    model.attach_encoder(encoder_loc)
    model.attach_templates(dataloader, template_size=32)
    model.save_to_disk(model_path(f"abstract2gene_lpb_{2**n}"))

    log(f"Final results (lpb = {2**n}):")
    log(f"  Test loss: {results["test_loss"][-1]}")
    log(f"  Test accuracy: {results["test_accuracy"][-1]}")

log(f"Number of genes: {len(model.templates.values)}")
