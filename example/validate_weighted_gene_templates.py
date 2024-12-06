import matplotlib.pyplot as plt
import numpy as np
import pubnet
import scipy as sp
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

model: a2g.model.Model = a2g.model.ModelNoWeights(
    features, labels, symbols, 42, name="noweights"
)
model.train(ndims=10, learning_rate=0.002, learning_decay=0.1)
model.test(max_num_tests=MAX_GENE_TESTS)

model = a2g.model.ModelJax(
    features,
    labels,
    symbols,
    42,
    batch_size=64,
)

# Coarse dimension search
for d in range(1, 20, 2):
    model.reset_rng()
    model.set_name(f"random_weights_{d}_dims")
    model.train(ndims=d, learning_rate=1e-4)
    model.test(max_num_tests=MAX_GENE_TESTS)

# Finer dimension search based on coarse results
for d in range(20, 110, 10):
    model.reset_rng()
    model.set_name(f"random_weights_{d}_dims")
    model.train(ndims=d, learning_rate=1e-4)
    model.test(max_num_tests=MAX_GENE_TESTS)

rng = default_rng(15)
features, labels, label_ids = a2g.processing.net2dataset(
    net, min_label_occurrences=N_SAMPLES, remove_baseline=True
)
symbols = net.get_node("Gene").feature_vector("GeneSymbol")[label_ids]
mix_idx = rng.permutation(np.arange(features.shape[0]))
features = features[mix_idx, :]
labels = labels[mix_idx, :]

model = a2g.model.ModelNoWeights(
    features, labels, symbols, 42, name="noweights_baseline_removed"
)
model.train(ndims=10, learning_rate=0.002, learning_decay=0.1)
model.test(max_num_tests=MAX_GENE_TESTS)

# Plot weights
fig, ax = plt.subplots()
ranks = sp.stats.rankdata(model.weights, axis=0)
im = ax.imshow(ranks)

ax.set_aspect(aspect=ranks.shape[1] / ranks.shape[0])
fig.tight_layout()
plt.savefig("figures/weights.png")
