"""Test the abstract2gene model against PubTator3 and pubmed annotations."""

import os

import datasets
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import abstract2gene as a2g
import example._config as cfg
from abstract2gene.dataset import mutators

FIGDIR = "figures/model_comparison"

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)


def plot(df: pd.DataFrame, filename: str | None = None, x_name: str = "Gene"):
    from plotnine import (
        aes,
        element_text,
        geom_errorbar,
        geom_point,
        ggplot,
        labs,
        position_dodge,
        position_jitterdodge,
        theme,
    )

    def stderr(x):
        return np.std(x) / np.sqrt(x.shape[0])

    n_symbols = len(df["symbol"].unique())
    metrics = df.groupby(["tag", "symbol"], as_index=False)["score"].agg(
        ["mean", "std", stderr]
    )

    jitter = position_jitterdodge(
        jitter_width=0.2, dodge_width=0.8, random_state=0
    )
    dodge = position_dodge(width=0.8)
    p = (
        ggplot(df, aes(x="symbol", color="tag"))
        + geom_point(aes(y="score"), position=jitter, size=1, alpha=0.4)
        + geom_errorbar(
            aes(
                y="mean",
                ymin="mean - (1.95 * stderr)",
                ymax="mean + (1.95 * stderr)",
                fill="tag",
            ),
            data=metrics,
            color="black",
            position=dodge,
            width=0.8,
            size=0.6,
        )
        + labs(y="Similarity", x=x_name, color="Tag")
        + theme(
            axis_text_x=element_text(
                rotation=45 if n_symbols > 10 else 0,
                ha="right" if n_symbols > 10 else "center",
                rotation_mode="anchor" if n_symbols > 10 else "default",
            ),
            text=element_text(family=cfg.font_family, size=cfg.font_size),
        )
    )
    if filename:
        p.save(
            os.path.join(FIGDIR, filename),
            width=cfg.fig_width,
            height=cfg.fig_height,
        )
    else:
        p.show()


model_results: dict[str, pd.DataFrame] = {}
dataset = datasets.load_dataset(
    f"{cfg.hf_user}/pubtator3_abstracts", data_files=cfg.TEST_FILES
)["train"]
dataset = mutators.translate_to_human_orthologs(dataset, cfg.max_cpu)
symbols = mutators.get_gene_symbols(dataset)

for name in [f"abstract2gene_lpb_{2**n}" for n in range(1, 9)]:
    model = a2g.model.load_from_disk(name)
    df = a2g.model.test(
        model, dataset, "gene", symbols=symbols, n_samples=30_000
    )
    model_results[name] = df
    plot(df, f"{name}.{cfg.figure_ext}")

for name, results in model_results.items():
    descriptor = name.split("_")[-1]
    results.symbol = descriptor

df_agg = pd.concat(model_results)
categories = sorted(df_agg.symbol.unique(), key=int)
cat_type = CategoricalDtype(categories=categories, ordered=True)
df_agg["symbol"] = df_agg["symbol"].astype(cat_type)

plot(df_agg, f"model_comparison.{cfg.figure_ext}", x_name="Labels Per Batch")

## Not enough pubmed genes to perform
# dataset = mutators.attach_pubmed_genes(dataset, "gene2pubmed", max_cpu=10)
# df = a2g.model.test(
#     model, dataset, "gene2pubmed", symbols=symbols, n_samples=50_000
# )
# a2g.model.plot(df, os.path.join(FIGDIR, "multi_layer_pubmed_labels.png"))
