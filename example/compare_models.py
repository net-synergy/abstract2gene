import matplotlib.pyplot as plt
import numpy as np
import pubnet
import scipy as sp

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

data = a2g.dataset.net2dataset(
    net,
    min_occurrences=N_SAMPLES,
    label_name="GeneSymbol",
    feature_name="PMID",
    seed=42,
    batch_size=64,
    template_size=32,
)

model: a2g.model.Model = a2g.model.ModelNoWeights(name="noweights")
a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS, save_results=False)

model = a2g.model.ModelSingleLayer(name="", seed=12, n_dims=20)

# Coarse dimension search
for d in range(1, 20, 2):
    data.reset_rng()
    model.name = f"random_weights_{d}_dims"
    a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
    a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)

# Finer dimension search based on coarse results
for d in range(20, 110, 10):
    data.reset_rng()
    model.name = f"random_weights_{d}_dims"
    a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
    a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)


dims = (data.n_features, 64)
model = a2g.model.ModelMultiLayer(
    name="multi", seed=20, dims=dims, dropout=0.2
)
a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)

data = a2g.dataset.net2dataset(
    net,
    min_occurrences=100,
    label_name="GeneSymbol",
    feature_name="PMID",
    seed=42,
    batch_size=32,
    template_size=32,
    axis=0,
)
dims = (data.n_features, 64)
model = a2g.model.ModelWidth(name="wide", seed=20, dims=dims, dropout=0.2)
a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)

data = a2g.dataset.net2dataset(
    net, min_occurrences=N_SAMPLES, remove_baseline=True
)

model = a2g.model.ModelNoWeights(name="noweights_baseline_removed")
a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)

# Plot weights
fig, ax = plt.subplots()
ranks = sp.stats.rankdata(model.params["w"], axis=0)
im = ax.imshow(ranks)
ax.set_aspect(aspect=ranks.shape[1] / ranks.shape[0])
fig.tight_layout()
plt.savefig("figures/weights.png")
