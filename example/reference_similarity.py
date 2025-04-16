import os
import sys

import datasets
import numpy as np
import pandas as pd
import plotnine as p9
import speakeasy2 as se2

import abstract2gene as a2g
import example._config as cfg

N_PUBLICATIONS = 10
FIGDIR = "figures/reference_similarities/"
MODEL = "a2g_768dim_per_batch_4"
k = 5

seed = cfg.seeds["reference_similarity"]

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

model = a2g.model.load_from_disk(MODEL)
dataset = datasets.load_dataset("dconnell/pubtator3_abstracts")["train"]

rng = np.random.default_rng(seed=seed)
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

graph = se2.knn_graph(corr, k)
clusters = se2.cluster(graph, subcluster=2, seed=seed + 1)
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
    + p9.labs(x="Count", y="Cluster", color="Cited by", fill="Cited by")
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
    )
)
p.save(
    os.path.join(FIGDIR, f"cluster_dist_{MODEL}_{k}.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height,
)

p = (
    p9.ggplot(
        communities,
        p9.aes(x="cluster", fill="cited_by", color="cited_by", alpha="type"),
    )
    + p9.scale_alpha_discrete(range=(0.5, 1))
    + p9.geom_bar()
    + p9.coord_flip()
    + p9.labs(
        x="Count",
        y="Cluster",
        color="Cited by",
        fill="Cited by",
        # Otherwise "Type" gets cutoff for some reason.
        alpha=r"-\\Type",
    )
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
    )
)
p.save(
    os.path.join(
        FIGDIR,
        f"cluster_dist_highlight_molecular_{MODEL}_{k}.{cfg.figure_ext}",
    ),
    width=cfg.fig_width,
    height=cfg.fig_height,
)

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
thresh = 0.1
predictions = [
    {"parent_id": pub_id, "pmid": str(pmid), "gene": gene, "prediction": pred}
    for pub_id, (pub_predictions, pmid) in enumerate(zip(regression, pmids))
    for gene, pred in enumerate(pub_predictions)
    if pred > thresh
]

parent_gene_predictions = pd.DataFrame(predictions)
p = (
    p9.ggplot(
        parent_gene_predictions,
        p9.aes(x="pmid", y="prediction", fill="pmid", group="gene"),
    )
    + p9.geom_col(stat="identity", position="dodge", show_legend=False)
    + p9.labs(x="PMID", y=r"Gene predictions $> " + str(thresh) + r"$")
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        axis_text_x=p9.element_text(
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        ),
    )
)
p.save(
    os.path.join(FIGDIR, f"parent_gene_dist_{MODEL}.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height,
)
