"""Train the final abstract2gene prediction model.

The abstract2gene model consists of an embedding model to convert the title +
abstract into a fixed-width vector and a single dense layer. The dense layer is
trained here (weights of the embedding model are fixed).

The dataset abstract2gene is trained on is masked for genes + disease
annotations to prevent the model from using that data in its predictions. The
model is trained under multiple labels per batch conditions, resulting in one
model for each "label per batch" value.
"""

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

EXPERIMENT = "train_abstract2gene"
seed = cfg.seeds[EXPERIMENT]
set_log(EXPERIMENT)

encoder_loc = f"{cfg.hf_user}/{cfg.encoder["remote_name"]}"
encoder = SentenceTransformer(encoder_loc)

dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts",
    data_files=cfg.A2G_TRAIN_FILES,
)["train"]

log("Converting genes to human orthologs:")
log("  Before conversion:")
log(f"    {len(dataset.features["gene"].feature.names)} unique genes")
log(f"    {len([g for gs in dataset["gene"] for g in gs])} total genes")
dataset = mutators.translate_to_human_orthologs(dataset, cfg.max_cpu)
log("  After conversion:")
log(f"    {len(dataset.features["gene"].feature.names)} unique genes")
log(f"    {len([g for gs in dataset["gene"] for g in gs])} total genes")
log("")

genes = np.bincount(jax.tree.leaves(dataset["gene"]))
mask = genes > cfg.template_size
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
    model.attach_templates(dataloader, template_size=cfg.template_size)
    model.save_to_disk(model_path(f"abstract2gene_lpb_{2**n}"))

    log(f"Final results (lpb = {2**n}):")
    log(f"  Test loss: {results["test_loss"][-1]}")
    log(f"  Test accuracy: {results["test_accuracy"][-1]}")

log(f"Number of genes: {len(model.templates.values)}")
