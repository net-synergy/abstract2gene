import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import speakeasy2 as se2

import abstract2gene as a2g

SEED = 10
N_PUBLICATIONS = 10
MODEL = "abstract2gene"

model = a2g.model.load_from_disk(MODEL)
dataset = datasets.load_dataset("dconnell/pubtator3_abstracts")["train"]

rng = np.random.default_rng(seed=SEED)
parent_publications = [
    int(pub)
    for pub in rng.integers(0, len(dataset), N_PUBLICATIONS * 10)
    if len(dataset[int(pub)]["reference"]) > 0
    and len(dataset[int(pub)]["gene"]) > 0
][:N_PUBLICATIONS]

reference_lists = dataset[parent_publications]["reference"]

dataset = dataset.with_format(
    "numpy", columns=["pmid"], output_all_columns=True
)
pmids = dataset["pmid"]
idx = np.arange(len(pmids))
sort_idx = np.argsort(pmids)

idx = idx[sort_idx]
pmids = pmids[sort_idx]

indices = [
    [int(idx[np.searchsorted(pmids, ref)]) for ref in ref_list]
    for ref_list in reference_lists
]

# In case any reference IDs are not in the dataset. A check for the current
# random seed shows all this analysis' references are in the dataset.
#
# Can compare number of references before and after:
#   print([len(ref_list) for ref_list in reference_lists])
indices = [
    [i for i, ref in zip(ref_indices, ref_list) if dataset[i]["pmid"] == ref]
    for ref_indices, ref_list in zip(indices, reference_lists)
]
ground_truth = [
    i for i, ref_list in enumerate(reference_lists) for _ in ref_list
]
ref_ds = dataset.select([i for ref_indices in indices for i in ref_indices])
inputs = [
    title + "[SEP]" + abstract
    for title, abstract in zip(ref_ds["title"], ref_ds["abstract"])
]

regression = np.array(model.predict(inputs))
regression = regression / (np.linalg.norm(regression, axis=1, keepdims=True))
corr = regression @ regression.T
np.fill_diagonal(corr, 0)

graph = se2.knn_graph(corr, 10, is_weighted=True)
clusters = se2.cluster(graph, subcluster=2, seed=SEED + 1)
ordering = se2.order_nodes(corr, clusters)

comm_dict = [
    {
        "cluster": i,
        "cited_by": str(
            dataset[parent_publications[ground_truth[member]]]["pmid"]
        ),
        "type": "molecular" if len(ref_ds[member]["gene"]) else "behavioral",
    }
    for i, cluster in enumerate(clusters[0])
    for member in cluster
]

count = 0
cluster = 0
sc_i = 0
for subcluster in clusters[1]:
    for _ in subcluster:
        if comm_dict[count]["cluster"] != cluster:
            cluster += 1
            sc_i = 0
        comm_dict[count]["subcluster"] = sc_i
        count += 1
    sc_i += 1

communities = pd.DataFrame(comm_dict)

p = (
    p9.ggplot(
        communities, p9.aes(x="cluster", fill="cited_by", color="cited_by")
    )
    + p9.geom_bar()
    + p9.coord_flip()
)
p.save("figures/reference_similarities/cluster_dist.svg")

p = (
    p9.ggplot(
        communities,
        p9.aes(x="subcluster", fill="cited_by", color="cited_by"),
    )
    + p9.facet_wrap("~cluster")
    + p9.geom_bar()
    + p9.coord_flip()
)
p.save("figures/reference_similarities/cluster_dist_level_2.svg")

p = (
    p9.ggplot(
        communities,
        p9.aes(x="cluster", fill="cited_by", color="cited_by", alpha="type"),
    )
    + p9.scale_alpha_discrete(range=(0.5, 1))
    + p9.geom_bar()
    + p9.coord_flip()
)
p.save("figures/reference_similarities/cluster_dist_highlight_molecular.svg")

## Analyze parent publications
inputs = [
    title + "[SEP]" + abstract
    for title, abstract in zip(
        dataset[parent_publications]["title"],
        dataset[parent_publications]["abstract"],
    )
]

regression = np.array(model.predict(inputs))

pmids = dataset[parent_publications]["pmid"]
predictions = [
    {"parent_id": pub_id, "pmid": str(pmid), "gene": gene, "prediction": pred}
    for pub_id, (pub_predictions, pmid) in enumerate(zip(regression, pmids))
    for gene, pred in enumerate(pub_predictions)
    if pred > 0.1
]

parent_gene_predictions = pd.DataFrame(predictions)
p = (
    p9.ggplot(
        parent_gene_predictions,
        p9.aes(x="pmid", y="prediction", fill="pmid", group="gene"),
    )
    + p9.geom_col(stat="identity", position="dodge", show_legend=False)
    + p9.theme(axis_text_x=p9.element_text(rotation=10))
)
p.save("figures/reference_similarities/parent_gene_dist.svg")

## Extension to behavioral publications
rng = np.random.default_rng(seed=SEED + 2)
parent_publications = [
    int(pub)
    for pub in rng.integers(0, len(dataset), 10000)
    if len(dataset[int(pub)]["reference"]) > 0
    and len(dataset[int(pub)]["gene"]) > 0
]

reference_lists = dataset[parent_publications]["reference"]

dataset = dataset.with_format(
    "numpy", columns=["pmid"], output_all_columns=True
)
pmids = dataset["pmid"]
idx = np.arange(len(pmids))
sort_idx = np.argsort(pmids)

idx = idx[sort_idx]
pmids = pmids[sort_idx]

indices = [
    [int(idx[np.searchsorted(pmids, ref)]) for ref in ref_list]
    for ref_list in reference_lists
]

indices = [
    [
        i
        for i, ref in zip(ref_indices, ref_list)
        if ((dataset[i]["pmid"] == ref) and (len(dataset[i]["gene"]) == 0))
    ]
    for ref_indices, ref_list in zip(indices, reference_lists)
]

selected_references = [
    (
        int(idx[np.searchsorted(pmids, parent)]),
        int(rng.choice(ref_indices, 1)[0]),
    )
    for parent, ref_indices in zip(parent_publications, indices)
    if len(ref_indices) > 0
]

random_behavioral_study: list[int] = []
while len(random_behavioral_study) < len(selected_references):
    ref = int(rng.integers(0, len(dataset), 1)[0])
    if (ref not in random_behavioral_study) and (
        len(dataset[ref]["gene"]) == 0
    ):
        random_behavioral_study.append(ref)

inputs = [
    tuple(
        dataset[ref]["title"] + "[SEP]" + dataset[ref]["abstract"]
        for ref in (parent, citation, random)
    )
    for random, (parent, citation) in zip(
        random_behavioral_study, selected_references
    )
]

probabilities = [np.array(model.predict(abstracts)) for abstracts in inputs]
parent = np.hstack([[i, i] for i in range(len(inputs))])
group = np.hstack([["reference", "random"] for i in range(len(inputs))])
corr = np.hstack(
    [
        [
            (ref_set[0, :] @ ref_set[1, :]) / (ref_set[0, :] @ ref_set[0, :]),
            (ref_set[0, :] @ ref_set[2, :]) / (ref_set[0, :] @ ref_set[0, :]),
        ]
        for ref_set in probabilities
    ]
)

corrs = pd.DataFrame({"parent": parent, "group": group, "correlation": corr})

p = p9.ggplot(corrs, p9.aes(x="group", y="correlation")) + p9.geom_jitter()
p.save("figures/reference_similarities/correlation.svg")

selected_genes = [np.argmax(p_set[0]) for p_set in probabilities]
selected_probabilities = np.vstack(
    [
        (p_set[0][gene], p_set[1][gene], p_set[2][gene])
        for gene, p_set in zip(selected_genes, probabilities)
    ]
)

fig, ax = plt.subplots()
sort_idx = np.argsort(selected_probabilities[:, 0])
ax.plot(
    selected_probabilities[sort_idx, 0],
    selected_probabilities[sort_idx, 1],
    label="citation",
)
ax.plot(
    selected_probabilities[sort_idx, 0],
    selected_probabilities[sort_idx, 2],
    label="random",
)
ax.legend(loc="upper left")
ax.set_xlabel("parent prediction")
ax.set_ylabel("reference prediction")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.savefig("figures/reference_similarities/correlation.svg")
plt.close()
