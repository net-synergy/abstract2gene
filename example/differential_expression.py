import os

import datasets
import numpy as np
import pandas as pd
import plotnine as p9
import scipy.stats as stats

import abstract2gene as a2g
from abstract2gene.data import PubmedDownloader
from abstract2gene.dataset import mutators
from example import config as cfg

## Private data
TRANSCRIPTOME_PATH = "/disk4/data/adBulkTranscriptome/"

# Look for genes with a p-value below ALPHA but show at most MAX_GENES
ALPHA = 0.05
MAX_GENES = 50
FIGDIR = "figures/differential_expression/"

seed = cfg.seeds["differential_expression"]
if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts",
    data_files=cfg.AD_DE_FILES,
)["train"]

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

# When sorted, the below iloc key are rows containing the ENSG identifiers.
ens2entrez = (
    pd.read_table(files[0], usecols=["GeneID", "Ensembl_gene_identifier"])
    .sort_values("Ensembl_gene_identifier")
    .iloc[3_156_346:3_227_727]
    .drop_duplicates()
)
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


dataset = dataset.filter(lambda example: len(example["gene"]) > 0)
ad_mask = np.fromiter(
    (is_ad_abstract(abstract) for abstract in dataset["abstract"]),
    dtype=np.bool,
)

rng = np.random.default_rng(seed=seed)
n_samples = ad_mask.sum()
publications_ad = np.arange(len(ad_mask))[ad_mask]
publications_other = rng.choice(
    np.arange(len(ad_mask))[np.logical_not(ad_mask)], n_samples, replace=False
)


## Compare publication gene predictions
def inputs(index: np.ndarray) -> list[str]:
    return [
        title + "[SEP]" + abstract
        for title, abstract in zip(
            dataset[index]["title"], dataset[index]["abstract"]
        )
    ]


model = a2g.model.load_from_disk("a2g_768dim_per_batch_2")
np.isin(model.templates.indices, de_idx)
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
de_model_idx, de_dataset_idx = zip(*de_dataset2model)
symbols = [symbols[i] for i in de_dataset_idx]

for name in [f"a2g_768dim_per_batch_{2**n}" for n in range(1, 7)]:
    model = a2g.model.load_from_disk(name)
    predictions = pd.DataFrame(
        {
            "tag": np.array(
                (
                    (["AD"] * n_samples * n_genes)
                    + (["other"] * n_samples * n_genes)
                )
                * 2
            ),
            "label": np.array(
                (["a2g"] * n_samples * n_genes * 2)
                + (["pubtator3"] * n_samples * n_genes * 2)
            ),
            "gene": np.array(symbols * n_samples * 4),
            "prediction": (
                np.hstack(
                    (
                        np.array(model.predict(inputs(publications_ad)))[
                            :, de_model_idx
                        ].flatten(),
                        np.array(model.predict(inputs(publications_other)))[
                            :, de_model_idx
                        ].flatten(),
                        np.array(
                            [
                                gene in labels
                                for labels in dataset[
                                    np.hstack(
                                        (publications_ad, publications_other)
                                    )
                                ]["gene"]
                                for gene in de_dataset_idx
                            ]
                        ),
                    )
                )
                > 0.5
            ),
        }
    )

    ## Hypothesis testing (One-sided proportion test)
    sample_params = predictions.groupby(["tag", "gene", "label"]).agg(
        ["sum", "mean"]
    )
    sample_params.columns = [
        "prop" if col == "mean" else col for _, col in sample_params.columns
    ]

    def proportion_test(sample_params):
        prop = "prop"
        n = n_samples
        x = "sum"
        alpha = 0.05
        z_crit = stats.norm.ppf(1 - (alpha / (n_genes)))
        sample_prop = (
            sample_params.loc["AD"][x] + sample_params.loc["other"][x]
        ) / (2 * n)
        z_score = (
            sample_params.loc["AD"][prop] - sample_params.loc["other"][prop]
        )
        z_score /= np.sqrt((2 * sample_prop * (1 - sample_prop)) / n)

        return (z_score > z_crit).array

    significant = proportion_test(sample_params)
    sample_params = sample_params.reset_index()
    sample_params = sample_params.pivot(
        index=["label", "gene"], columns="tag", values=["prop"]
    )
    sample_params.columns = [
        f"{col[0]}_{col[1]}" for col in sample_params.columns
    ]

    # Need reshape trick because during the pivot we switch hierarchy from
    # gene>label to label>gene
    sample_params["significant"] = significant.reshape((-1, 2)).T.reshape((-1))
    mask = np.logical_or(
        np.logical_or(
            sample_params.loc["a2g", "prop_AD"] > 1e-2,
            sample_params.loc["a2g", "prop_other"] > 1e-2,
        ),
        np.logical_or(
            sample_params.loc["pubtator3", "prop_AD"] > 1e-2,
            sample_params.loc["pubtator3", "prop_other"] > 1e-2,
        ),
    )
    sample_params = sample_params.reset_index()
    sample_params = sample_params[np.hstack((mask, mask))]

    dodge_col = p9.position_dodge(width=0.4)
    p = (
        p9.ggplot(
            sample_params.reset_index(),
            p9.aes(x="gene", xend="gene", color="label", alpha="significant"),
        )
        + p9.geom_point(p9.aes(y="prop_AD"), size=0.2, position=dodge_col)
        + p9.geom_segment(
            p9.aes(y="prop_other", yend="prop_AD"),
            position=dodge_col,
        )
        + p9.scale_alpha_discrete(range=(0.35, 1))
        + p9.lims(y=(0, 0.5))
        + p9.labs(
            y="Proportion labeled", x="Gene", alpha="Significant difference"
        )
        + p9.theme(axis_text_x=p9.element_text(rotation=90))
    )
    p.save(os.path.join(FIGDIR, f"bernoulli_{name}.{cfg.figure_ext}"))

    p = (
        p9.ggplot(
            sample_params.reset_index(),
            p9.aes(x="gene", xend="gene", color="label", alpha="significant"),
        )
        + p9.geom_point(p9.aes(y="prop_AD"), size=0.2, position=dodge_col)
        + p9.geom_segment(
            p9.aes(y="prop_other", yend="prop_AD"),
            position=dodge_col,
        )
        + p9.scale_alpha_discrete(range=(0.35, 1))
        + p9.lims(y=(0, 0.5))
        + p9.scale_y_log10()
        + p9.labs(
            y="Proportion labeled", x="Gene", alpha="Significant difference"
        )
        + p9.theme(axis_text_x=p9.element_text(rotation=90))
    )
    p.save(os.path.join(FIGDIR, f"bernoulli_{name}_logy.{cfg.figure_ext}"))
