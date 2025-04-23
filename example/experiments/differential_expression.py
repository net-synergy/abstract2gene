"""Differential expression experiment.

Determine if the model labels genes differentially expression in Alzheimer's
disease patients more often in publications related to Alzheimer's disease than
in the general publication pool. Compares results when using molecular
publications (those with PubTator3 gene annotations) to behavioral publications
(those without any PubTator3 gene annotations).

Uses relative risk to determine if a publication having Alzheimer's disease
as a topic is associated with increased likelihood of also being related to a
given gene.

This analysis requires restricted data and therefore cannot be run by everyone.
"""

import os

import datasets
import numpy as np
import pandas as pd
import plotnine as p9
import scipy.stats as stats
from pandas.api.types import CategoricalDtype

import abstract2gene as a2g
import example._config as cfg
from abstract2gene.data import PubmedDownloader
from abstract2gene.dataset import mutators
from example._logging import log, set_log

## Private data
TRANSCRIPTOME_PATH = "/disk4/data/adBulkTranscriptome/"

# Look for genes with a p-value below ALPHA
ALPHA = 0.05

EXPERIMENT = "differential_expression"
FIGDIR = f"figures/{EXPERIMENT}/"

seed = cfg.seeds[EXPERIMENT]
set_log(EXPERIMENT)

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts",
    data_files=cfg.AD_DE_FILES,
)["train"]

ds_typed = {
    "molecular": dataset.filter(lambda example: len(example["gene"]) > 0),
    "behavioral": dataset.filter(lambda example: len(example["gene"]) == 0),
}

## Calculate differential gene expression between AD and NCI participants.

# samples x genes
counts = np.array(
    pd.read_csv(
        os.path.join(TRANSCRIPTOME_PATH, "bulkExpression.csv"),
        usecols=["counts"],
    )["counts"]
).reshape(-1, 17306)

counts = (counts - counts.mean(axis=0, keepdims=True)) / counts.std(
    axis=0, keepdims=True
)

samples = pd.read_csv(os.path.join(TRANSCRIPTOME_PATH, "bulkSamples.csv"))
ensemble_genes = pd.read_csv(os.path.join(TRANSCRIPTOME_PATH, "bulkGenes.csv"))

mf_mask = samples["area"] == "MF"
counts = counts[mf_mask, :]
samples = samples[mf_mask]
dup_genes = ensemble_genes.duplicated()
counts = counts[:, np.logical_not(dup_genes)]
ensemble_genes = ensemble_genes[np.logical_not(dup_genes)]

labels = pd.read_table(
    os.path.join(TRANSCRIPTOME_PATH, "projid_ad_labels.tsv"),
    header=0,
    names=["projId", "cogdx"],
    usecols=["projId", "cogdx"],
)
samples = samples.merge(labels, how="left", on="projId")
label_mask = np.logical_not(pd.isna(samples["cogdx"]))
samples = samples[label_mask]
samples["cogdx"] = samples["cogdx"].astype(int)
counts = counts[label_mask, :]

ad_mask = samples["cogdx"] == 4
nci_mask = samples["cogdx"] == 1

counts_ad = counts[ad_mask, :]
counts_nci = counts[nci_mask, :]

std_pool = np.sqrt(
    (
        ((counts_ad.shape[0] - 1) * counts_ad.var(axis=0))
        + ((counts_nci.shape[0] - 1) * counts_nci.var(axis=0))
    )
    / (counts_ad.shape[0] + counts_nci.shape[0] - 2)
)
z_scores = abs(counts_ad.mean(axis=0) - counts_nci.mean(axis=0)) / (
    std_pool * (np.sqrt((1 / counts_ad.shape[0]) + (1 / counts_nci.shape[0])))
)
z_crit = stats.norm.ppf(1 - (ALPHA / (2 * counts.shape[1])))

de_genes = ensemble_genes[z_scores > z_crit].copy()
de_genes["scores"] = z_scores[z_scores > z_crit]

## Get ensemble -> entrez gene ID map since the dataset uses entrez IDs.
pubmed_downloader = PubmedDownloader()
pubmed_downloader.files = ["gene2ensembl.gz"]
files = pubmed_downloader.download()

ens2entrez = pd.read_table(
    files[0], usecols=["GeneID", "Ensembl_gene_identifier"]
).sort_values("Ensembl_gene_identifier")

ens2entrez = ens2entrez[
    ens2entrez.Ensembl_gene_identifier.map(lambda x: x.startswith("ENSG"))
].drop_duplicates()

de_genes = de_genes.merge(
    ens2entrez,
    how="left",
    right_on="Ensembl_gene_identifier",
    left_on="ensembleId",
)[["GeneID", "scores"]]
de_genes = de_genes[np.logical_not(pd.isna(de_genes["GeneID"]))]
de_genes["GeneID"] = de_genes["GeneID"].astype(int)
de_genes = de_genes.sort_values("scores")

symbols = mutators.get_gene_symbols(dataset)
geneid2idx = dataset.features["gene"].feature.str2int
known_genes = dataset.features["gene"].feature.names
de_genes = de_genes[np.isin(de_genes["GeneID"], known_genes)]

de_idx = [geneid2idx(str(gene)) for gene in de_genes["GeneID"]]


## Find Alzheimer's Disease related publications
def is_ad_abstract(abstract: str) -> bool:
    """Test if abstract contains the word Alzheimer."""
    return "alzheimer" in abstract.lower()


def inputs(dataset: datasets.Dataset, index: np.ndarray) -> list[str]:
    return [
        title + model.sep_token + abstract
        for title, abstract in zip(
            dataset[index]["title"], dataset[index]["abstract"]
        )
    ]


def organize_predictions(
    preds_ad: np.ndarray, preds_other: np.ndarray, label: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tag": np.array(
                (["AD"] * preds_ad.shape[0])
                + (["other"] * preds_other.shape[0])
            ),
            "label": [label] * (preds_ad.shape[0] + preds_other.shape[0]),
            "gene": np.array(symbols * (2 * n_samples)),
            "prediction": np.hstack([preds_ad, preds_other]) > 0.5,
        }
    )


# Need to load up any model to get the template indices.
model = a2g.model.load_from_disk("a2g_768dim_per_batch_2")

de_dataset2model = [
    (
        i,
        de_idx[
            np.arange(len(de_idx))[model.templates.indices[i] == de_idx][0]
        ],
    )
    for i, tf in enumerate(np.isin(model.templates.indices, de_idx))
    if tf
]

n_genes = len(de_dataset2model)
log(f"DE genes in model: {n_genes}")
de_model_idx, de_dataset_idx = zip(*de_dataset2model)
symbols = [symbols[i] for i in de_dataset_idx]

rng = np.random.default_rng(seed=seed)
for k, ds in ds_typed.items():
    ad_mask = np.fromiter(
        (is_ad_abstract(abstract) for abstract in ds["abstract"]),
        dtype=np.bool,
    )

    n_samples = ad_mask.sum()
    log(f"AD samples ({k}): {n_samples}")

    publications_ad = np.arange(len(ad_mask))[ad_mask]
    publications_other = rng.choice(
        np.arange(len(ad_mask))[np.logical_not(ad_mask)],
        n_samples,
        replace=False,
    )

    model_predictions = []

    if k == "molecular":
        model_predictions = [
            organize_predictions(
                np.fromiter(
                    [
                        gene in labels
                        for labels in ds[publications_ad]["gene"]
                        for gene in de_dataset_idx
                    ],
                    dtype=np.bool,
                ),
                np.fromiter(
                    [
                        gene in labels
                        for labels in ds[publications_other]["gene"]
                        for gene in de_dataset_idx
                    ],
                    dtype=np.bool,
                ),
                "PubTator3",
            )
        ]

    for n in range(1, 7):
        lpb = 2**n
        name = f"a2g_768dim_per_batch_{lpb}"
        model = a2g.model.load_from_disk(name)

        model_predictions.append(
            organize_predictions(
                np.array(model.predict(inputs(ds, publications_ad)))[
                    :, de_model_idx
                ].flatten(),
                np.array(model.predict(inputs(ds, publications_other)))[
                    :, de_model_idx
                ].flatten(),
                f"A2G {lpb}",
            )
        )

    predictions = pd.concat(model_predictions, ignore_index=True)

    ## Relative Risk
    def events(x):
        return x.sum() + 0.5

    def non_events(x):
        return np.logical_not(x).sum() + 0.5

    sample_params = predictions.groupby(["tag", "gene", "label"]).agg(
        [events, non_events]
    )
    sample_params.columns = [col for _, col in sample_params.columns]

    sample_params = sample_params.reset_index()
    sample_params = sample_params.pivot(
        index=["label", "gene"], columns="tag", values=["events", "non_events"]
    )
    sample_params.columns = [
        f"{col[0]}_{col[1]}" for col in sample_params.columns
    ]
    sample_params["relative_risk"] = (
        sample_params["events_AD"]
        * (sample_params["events_other"] + sample_params["non_events_other"])
    ) / (
        sample_params["events_other"]
        * (sample_params["events_AD"] + sample_params["non_events_AD"])
    )
    sample_params["log_RR"] = np.log10(sample_params["relative_risk"])
    sample_params["se_log_RR"] = np.sqrt(
        (
            sample_params["non_events_AD"]
            / (
                sample_params["events_AD"]
                * (sample_params["events_AD"] + sample_params["non_events_AD"])
            )
        )
        + (
            sample_params["non_events_other"]
            / (
                sample_params["events_other"]
                * (
                    sample_params["events_other"]
                    + sample_params["non_events_other"]
                )
            )
        )
    )

    z_crit = stats.norm.ppf(1 - (ALPHA / 2))
    sample_params["ci_low"] = sample_params["log_RR"] - (
        sample_params["se_log_RR"] * z_crit
    )
    sample_params["ci_high"] = sample_params["log_RR"] + (
        sample_params["se_log_RR"] * z_crit
    )

    mask = (sample_params["ci_low"].unstack().T > 0).any(axis=1)
    sample_params = sample_params.reset_index()
    sample_params = sample_params[
        np.hstack([mask] * len(sample_params.label.unique()))
    ]
    sample_params.loc[:, "alpha"] = sample_params["ci_low"] > 0

    categories = sample_params.label.unique()
    idx = categories != "PubTator3"
    categories[idx] = sorted(
        categories[idx], key=lambda x: int(x.split(" ")[-1])
    )

    cat_type = CategoricalDtype(categories=categories, ordered=True)
    sample_params["label"] = sample_params["label"].astype(cat_type)

    # For some reason RR > 1 gets cutoff unless padding with a couple lines
    # above when their are too many annotations.
    padding = r"-\vspace{2.5em}\\" if k == "molecular" else r"\noindent"

    dodge_col = p9.position_dodge(width=0.8)
    p = (
        p9.ggplot(
            sample_params,
            p9.aes(x="gene", y="log_RR", color="label", alpha="alpha"),
        )
        + p9.geom_hline(p9.aes(yintercept=0), color="black")
        + p9.geom_point(size=1, position=dodge_col)
        + p9.geom_errorbar(
            p9.aes(ymin="ci_low", ymax="ci_high"),
            width=0.3,
            size=0.3,
            position=dodge_col,
        )
        + p9.scale_alpha_discrete(range=(0.3, 1))
        + p9.scale_color_discrete()
        + p9.labs(
            y=r"$\log \left(\textrm{Relative Risk}\right)$",
            x="Gene",
            alpha=padding + r"$\textrm{RR} > 1$\\(95\% confidence)",
            color="Annotations",
        )
        # Based on the current results to keep axes locked between figures.
        + p9.ylim((-4, 5.5))
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
        os.path.join(FIGDIR, f"relative_risk_{k}.{cfg.figure_ext}"),
        width=cfg.fig_width,
        height=cfg.fig_height,
    )
