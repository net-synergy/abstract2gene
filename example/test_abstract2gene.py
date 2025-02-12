"""Test the abstract2gene model against PubTator3 and pubmed annotations."""

import datasets

import abstract2gene as a2g
from abstract2gene.dataset import mutators
from example import config as cfg

model = a2g.model.load_from_disk("abstract2gene")
dataset = datasets.load_dataset(
    "dconnell/pubtator3_abstracts", data_files=cfg.TEST_FILES
)["train"]

symbols = mutators.get_gene_symbols(dataset)
df = a2g.model.test(model, dataset, "gene", symbols=symbols, n_samples=10000)
a2g.model.plot(df, "multi_layer.svg")

dataset = mutators.attach_pubmed_genes(dataset, "gene2pubmed", max_cpu=1)
df = a2g.model.test(
    model, dataset, "gene2pubmed", symbols=symbols, n_samples=10000
)
a2g.model.plot(df, "multi_layer_pubmed_labels.svg")
