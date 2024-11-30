import os

import numpy as np
import pandas as pd
import pubnet
from numpy.random import default_rng
from sklearn.model_selection import LeaveOneOut

import abstract2gene as a2g
from abstract2gene.processing import net2dataset

GRAPH = "pubtator_test"

# Guarantees there is at least 32 publications left to create a template from
# for each gene when training.
N_SAMPLES = 64
MAX_GENE_TESTS = 100
RESULT_FILE = "results/template_validation.tsv"


net = pubnet.load_graph(
    GRAPH,
    ("Publication", "Gene"),
    (("Publication", "Gene"), ("Abstract_embedding", "Publication")),
)
genetic_pubs = np.unique(net.get_edge("Publication", "Gene")["Publication"])
net = net[genetic_pubs]
net.repack()

rng = default_rng(12)
features, labels, label_ids = net2dataset(net)
mix_idx = rng.choice(features.shape[0], features.shape[0], replace=False)
features = features[mix_idx, :]
labels = labels[mix_idx, :]

training_mask = a2g.model.split_genes(labels, train_size=0.8, seed=42)
model = a2g.model.train(features, labels[:, training_mask], 30, ndims=30)


def test_gene(predict, features, labels, symbol):
    features_gene = features[labels, :]
    # Make all tests have same number of samples.
    features_gene = features_gene[:N_SAMPLES, :]
    features_other = features[np.logical_not(labels), :]

    loo = LeaveOneOut()
    sim_within = np.zeros((N_SAMPLES))
    sim_between = np.zeros((N_SAMPLES))
    for i, (train_index, test_index) in enumerate(loo.split(features_gene)):
        template = features_gene[train_index, :].mean(axis=0, keepdims=True)
        sim_within[i] = predict(features_gene[test_index, :], template)[0, 0]
        sim_between[i] = predict(features_other[[i], :], template)[0, 0]

    gene_df = pd.DataFrame(
        {
            "gene": np.repeat(symbol, N_SAMPLES * 2),
            "group": np.concat(
                (
                    np.asarray("within").repeat(N_SAMPLES),
                    np.asarray("between").repeat(N_SAMPLES),
                )
            ),
            "similarity": np.concat((sim_within, sim_between)),
        }
    )

    header = not os.path.exists(RESULT_FILE)
    gene_df.to_csv(RESULT_FILE, sep="\t", index=False, mode="a", header=header)

    distance = sim_within.mean() - sim_between.mean()
    stderr = np.concat((sim_within, sim_between)).std(ddof=1)
    stderr /= N_SAMPLES * 2
    return distance / stderr


if os.path.exists(RESULT_FILE):
    os.unlink(RESULT_FILE)

test_labels = labels[:, np.logical_not(training_mask)]
n_tests = min(MAX_GENE_TESTS, test_labels.shape[1])
loss = 0
for i in range(n_tests):
    # TODO: model will be model.predict
    symbol = net.get_node("Gene").feature_vector("GeneSymbol")[label_ids[i]]
    loss += test_gene(model, features, test_labels[:, i], symbol)

print(f"average sample mean distance {loss / n_tests}")
