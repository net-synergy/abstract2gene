"""Test the abstract2gene model against PubTator3 and pubmed annotations."""

import datasets

import abstract2gene as a2g
from abstract2gene.dataset import mutators
from example import config as cfg

for name in [f"a2g_768dim_per_batch_{2**n}" for n in range(1, 7)]:
    model = a2g.model.load_from_disk(name)
    dataset = datasets.load_dataset(
        "dconnell/pubtator3_abstracts", data_files=cfg.TEST_FILES
    )["train"]

    symbols = mutators.get_gene_symbols(dataset)
    df = a2g.model.test(
        model, dataset, "gene", symbols=symbols, n_samples=30_000
    )
    a2g.model.plot(df, f"figures/model_comparison/{name}.png")

## Not enough pubmed genes to perform
# dataset = mutators.attach_pubmed_genes(dataset, "gene2pubmed", max_cpu=10)
# df = a2g.model.test(
#     model, dataset, "gene2pubmed", symbols=symbols, n_samples=50_000
# )
# a2g.model.plot(df, "figures/model_comparison/multi_layer_pubmed_labels.svg")
