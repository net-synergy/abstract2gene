import argparse
import os

import datasets
import numpy as np
import pandas as pd
import plotnine as p9
import speakeasy2 as se2
from pandas.api.types import CategoricalDtype

import abstract2gene as a2g
import example._config as cfg

EXPERIMENT = "reference_similarity"
FIGDIR = f"figures/{EXPERIMENT}"

seed = cfg.seeds[EXPERIMENT]

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

k = 5
lpb = 64
n_publications = 10
weighted = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        default=k,
        type=int,
        help="Number of edges in K nearest neighbors graph",
    )
    parser.add_argument(
        "--lpb",
        default=lpb,
        type=int,
        help="Labels per batch",
    )
    parser.add_argument(
        "--weighted",
        default=weighted,
        type=bool,
        help="Whether KNN should be weighted",
    )
    parser.add_argument(
        "--n_publications",
        default=n_publications,
        type=int,
        help="Number of parent publications",
    )

    args = parser.parse_args()

    lpb = args.lpb
    n_publications = args.n_publications
    weighted = args.weighted
    k = args.k

model_name = f"abstract2gene_lpb_{lpb}"
model = a2g.model.load_from_disk(model_name)
dataset = datasets.load_dataset(f"{cfg.hf_user}/pubtator3_abstracts")["train"]

rng = np.random.default_rng(seed=seed)
parent_publications = [
    int(pub)
    for pub in rng.integers(0, len(dataset), n_publications * 10)
    if len(dataset[int(pub)]["reference"]) > 0
    and len(dataset[int(pub)]["gene"]) > 0
][:n_publications]

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

# In case any reference IDs are not in the dataset.
#
# Can compare number of references before and after:
#   print([len(ref_list) for ref_list in reference_lists])
indices = [
    [i for i, ref in zip(ref_indices, ref_list) if dataset[i]["pmid"] == ref]
    for ref_indices, ref_list in zip(indices, reference_lists)
]
ground_truth = [i for i, ref_list in enumerate(indices) for _ in ref_list]
ref_ds = dataset.select([i for ref_indices in indices for i in ref_indices])

ref_inputs = [
    example["title"] + "[SEP]" + example["abstract"] for example in ref_ds
]

parent_inputs = [
    example["title"] + "[SEP]" + example["abstract"]
    for example in dataset.select(parent_publications)
]

ref_predictions = np.array(model.predict(ref_inputs))
ref_predictions = ref_predictions / (
    np.linalg.norm(ref_predictions, axis=1, keepdims=True)
)

parent_predictions = np.array(model.predict(parent_inputs))
parent_predictions = parent_predictions / (
    np.linalg.norm(parent_predictions, axis=1, keepdims=True)
)
cited_by = [
    dataset[parent_publications[cluster]]["pmid"] for cluster in ground_truth
]

corr = parent_predictions @ ref_predictions.T
corr -= corr.min()
corr /= corr.max()
df = pd.DataFrame(
    {
        "cited_by": cited_by,
        "closest": [
            dataset[parent_publications[cluster]]["pmid"]
            for cluster in corr.argmax(axis=0)
        ],
        "distance": 1 - corr.max(axis=0),
    }
)

df.cited_by = df.cited_by.transform(str)
df.closest = df.closest.transform(str)

categories = df.closest.unique()
categories = sorted(categories, reverse=True)
cat_type = CategoricalDtype(categories=categories, ordered=True)
df.closest = df.closest.astype(cat_type)

p = (
    p9.ggplot(df, p9.aes(x="distance", y="closest", color="cited_by"))
    + p9.geom_point(size=4)
    + p9.labs(x="Distance", y="Closest parent", color="Cited by")
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        axis_text_y=p9.element_text(
            rotation=-45,
            ha="right",
            rotation_mode="anchor",
        ),
    )
)
p.save(
    os.path.join(FIGDIR, f"closest_parent_{model_name}.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height,
)

corr = ref_predictions @ ref_predictions.T
np.fill_diagonal(corr, 0)

graph = se2.knn_graph(corr, k, is_weighted=weighted) if k else corr
clusters = se2.cluster(graph, subcluster=2, seed=seed + 1)

if not k:
    graph = se2.knn_graph(graph, 3)

k = str(k) + "_weighted" if weighted else str(k)
ordering = se2.order_nodes(graph, clusters)

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
    + p9.labs(x="Cluster", y="Count", color="Cited by", fill="Cited by")
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
    )
)
p.save(
    os.path.join(FIGDIR, f"cluster_dist_{model_name}_{k}.{cfg.figure_ext}"),
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
        x="Cluster",
        y="Count",
        color="Cited by",
        fill="Cited by",
        # Otherwise "Type" gets cutoff for some reason.
        alpha=r"-\\[2em]Type",
    )
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
    )
)
p.save(
    os.path.join(
        FIGDIR,
        f"cluster_dist_highlight_molecular_{model_name}_{k}.{cfg.figure_ext}",
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
    os.path.join(FIGDIR, f"parent_gene_dist_{model_name}.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height,
)
