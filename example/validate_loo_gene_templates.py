import os

import numpy as np
import pandas as pd
import pubnet
from numpy.random import default_rng
from sklearn.model_selection import LeaveOneOut

GRAPH = "pubtator_test"

N_SAMPLES = 50
MIN_GENE_OCCURANCE = N_SAMPLES
MAX_GENE_TESTS = 100
RESULT_FILE = "template_validation.tsv"


net = pubnet.load_graph(
    GRAPH,
    ("Publication", "Gene"),
    (("Publication", "Gene"), ("Abstract_embedding", "Publication")),
)
genetic_pubs = np.unique(net.get_edge("Publication", "Gene")["Publication"])

embeddings_edge = net.get_edge("Abstract_embedding", "Publication")
n_features = np.sum(embeddings_edge["Publication"] == 0)
embeddings = embeddings_edge.feature_vector("embedding").reshape(
    (-1, n_features)
)
embeddings = embeddings - np.mean(embeddings, axis=0, keepdims=True)
embeddings = embeddings[genetic_pubs, :]
embeddings = embeddings / np.reshape(
    np.linalg.norm(embeddings, axis=1), shape=(-1, 1)
)

net = net[genetic_pubs]
net.repack()
gene_edges = net.get_edge("Publication", "Gene")
gene_frequencies = np.unique_counts(gene_edges["Gene"])
locs = gene_frequencies.counts >= MIN_GENE_OCCURANCE
most_frequent_genes = gene_frequencies.values[locs]


def test_gene(gene_idx, embeddings, rng):
    gene_locs = gene_edges["Gene"] == gene_idx
    gene_pubs = gene_edges[gene_locs]["Publication"]
    other_pubs = gene_edges[np.logical_not(gene_locs)]["Publication"]

    rng.shuffle(gene_pubs, axis=0)
    rng.shuffle(other_pubs, axis=0)

    gene_embeddings = embeddings[gene_pubs, :]
    loo = LeaveOneOut()
    sim_within = np.zeros((N_SAMPLES))
    sim_between = np.zeros((N_SAMPLES))
    for i, (train_index, test_index) in enumerate(loo.split(gene_embeddings)):
        if i == N_SAMPLES:
            break

        template = gene_embeddings[train_index, :].mean(axis=0)
        sim_within[i] = np.dot(
            template, gene_embeddings[test_index, :].squeeze()
        )
        sim_between[i] = np.dot(
            template, embeddings[other_pubs[i], :].squeeze()
        )

    gene_name = net.get_node("Gene").loc(gene_idx).feature_vector("GeneSymbol")

    gene_df = pd.DataFrame(
        {
            "gene": np.repeat(gene_name, N_SAMPLES * 2),
            "group": np.concat(
                (
                    np.asarray("within").repeat(N_SAMPLES),
                    np.asarray("between").repeat(N_SAMPLES),
                )
            ),
            "similarity": np.concat((sim_within, sim_between)),
        }
    )

    if gene_idx == most_frequent_genes[0]:
        header = True
        mode = "w"
    else:
        header = False
        mode = "a"

    gene_df.to_csv(
        RESULT_FILE, sep="\t", index=False, mode=mode, header=header
    )


rng = default_rng(2390)
rng.shuffle(most_frequent_genes)
for i in range(min(MAX_GENE_TESTS, most_frequent_genes.shape[0])):
    test_gene(most_frequent_genes[i], embeddings, rng)
