import numpy as np
import pubnet
from numpy.random import default_rng

import abstract2gene as a2g

GRAPH = "pubtator_test"

# Guarantees there is at least 32 publications left to create a template from
# for each gene when training.
N_SAMPLES = 64
MAX_GENE_TESTS = 100

net = pubnet.load_graph(
    GRAPH,
    ("Publication", "Gene"),
    (("Publication", "Gene"), ("Abstract_embedding", "Publication")),
)
genetic_pubs = np.unique(net.get_edge("Publication", "Gene")["Publication"])
net = net[genetic_pubs]
net.repack()

rng = default_rng(15)
features, labels, label_ids = a2g.processing.net2dataset(
    net, min_label_occurrences=N_SAMPLES, remove_baseline=False
)
symbols = net.get_node("Gene").feature_vector("GeneSymbol")[label_ids]
mix_idx = rng.choice(features.shape[0], features.shape[0], replace=False)
features = features[mix_idx, :]
labels = labels[mix_idx, :]

model = a2g.model.ModelNoWeights(
    features, labels, symbols, 42, name="mse", train_test_val=(0.6, 0.3, 0.1)
)
model.train(ndims=10, learning_rate=0.002, learning_decay=0.1)
model.test(max_num_tests=MAX_GENE_TESTS)
