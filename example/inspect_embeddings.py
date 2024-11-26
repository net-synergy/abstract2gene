import matplotlib.pyplot as plt
import numpy as np
import pubnet
from numpy.random import default_rng

GRAPH = "pubtator_test"
N_SAMPLES = 100
MIN_GENE_OCCURANCE = N_SAMPLES
N_GENES = 50

net = pubnet.load_graph(
    GRAPH,
    ("Publication", "Gene"),
    (("Publication", "Gene"), ("Abstract_embedding", "Publication")),
)
# net.repack()

gene_edges = net.get_edge("Publication", "Gene")
gene_frequencies = np.unique_counts(gene_edges["Gene"])
frequent_genes = gene_frequencies.values[
    gene_frequencies.counts >= MIN_GENE_OCCURANCE
]

embeddings_edge = net.get_edge("Abstract_embedding", "Publication")
n_features = np.sum(embeddings_edge["Publication"] == 0)
embeddings = embeddings_edge.feature_vector("embedding").reshape(
    (-1, n_features)
)
# embeddings = embeddings - np.mean(embeddings, axis=0, keepdims=True)
embeddings = embeddings / np.reshape(
    np.linalg.norm(embeddings, axis=1), shape=(-1, 1)
)


def variance(embeddings, rng):
    sample = rng.permutation(embeddings, axis=0)[:N_SAMPLES]
    sample = sample[::2, :] * sample[1::2, :]
    sample_mean = np.mean(sample, axis=0, keepdims=True)
    sample_var = np.var(sample, axis=0, mean=sample_mean, ddof=1)

    return (sample_var, sample_mean.squeeze())


rng = default_rng(2254)
frequent_genes = frequent_genes[
    net.get_node("Gene").loc(frequent_genes).feature_vector("GeneSymbol")
    != None  # noqa: E711
]
rng.shuffle(frequent_genes)

genetic_pubs = np.unique(net.get_edge("Gene", "Publication")["Publication"])
genetic_locs = np.isin(net.get_node("Publication").index, genetic_pubs)
non_genetic_locs = np.logical_not(genetic_locs)

labels = ["Non-Genetic"]
means = np.zeros((N_GENES + 1, n_features))
within_var = np.zeros((N_GENES + 1, n_features))

within_var[0, :], means[0, :] = variance(embeddings[non_genetic_locs], rng)
for i in range(N_GENES):
    labels.append(
        net.get_node("Gene")
        .loc(frequent_genes[i])
        .feature_vector("GeneSymbol")[0]
        .split("|")[0]
    )
    gene_pubs = gene_edges[gene_edges["Gene"] == frequent_genes[i]][
        "Publication"
    ]
    within_var[i + 1, :], means[i + 1, :] = variance(
        embeddings[gene_pubs], rng
    )

between_var = np.var(means, axis=0, keepdims=True, ddof=1)
relative_stability = between_var / (within_var + between_var)

labels = [l if len(l) < 10 else l[:10] for l in labels]
fig, ax = plt.subplots()
im = ax.imshow(relative_stability.T)
ax.set_xlabel("Gene")
ax.set_ylabel("Feature")
ax.set_title("Proportion of total variance due to between group variance")

# ax.set_xticks(np.arange(len(labels)), labels=labels)
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax.set_aspect(aspect=51 / 768)
fig.tight_layout()
plt.savefig("figures/embeddings_stability.png")

average_stability = np.mean(relative_stability, axis=0)
fig, ax = plt.subplots()

ax.hist(average_stability, bins=25)
ax.set_xlabel(
    "Proportion of features total variance due to between group variance"
)
plt.savefig("figures/embeddings_variance_dist.png")

fig, ax = plt.subplots()
ax.hist(
    relative_stability[0, :] - np.mean(relative_stability[1:, :], axis=0),
    bins=20,
)
ax.set_xlabel(
    "Difference in genetic and non-genetic features relative between "
    + "group variance."
)
plt.savefig("figures/genetic_vs_non_variance.png")
