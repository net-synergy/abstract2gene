import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

import abstract2gene as a2g

SEED = 10
N_PUBLICATIONS = 10
MODEL = "abstract2gene"

model = a2g.model.load_from_disk(MODEL)
dataset = datasets.load_dataset("dconnell/pubtator3_abstracts")["train"]
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
    [
        dataset[ref]["title"] + "[SEP]" + dataset[ref]["abstract"]
        for ref in (parent, citation, random)
    ]
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
plt.savefig("figures/reference_similarities/selected_gene_correlation.svg")
plt.close()
