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
    net, min_label_occurrences=N_SAMPLES
)
symbols = net.get_node("Gene").feature_vector("GeneSymbol")[label_ids]
mix_idx = rng.permutation(np.arange(features.shape[0]))
features = features[mix_idx, :]
labels = labels[mix_idx, :]

model = a2g.model.ModelMaximizeDistance(
    features, labels, symbols, 42, name="mse"
)
model.train(ndims=768, learning_rate=4, learning_decay=0.01, dropout=0.0)
model.test(max_num_tests=MAX_GENE_TESTS)
