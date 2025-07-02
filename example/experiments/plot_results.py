"""Plot saved data.

Uses data collected in experiments to plot figures. Intended to make it easier
to experiment with visuals without having to rerun the entire experiment.
"""

import numpy as np
import pandas as pd
import plotnine as p9
from pandas.api.types import CategoricalDtype

import example._config as cfg


def stderr(x):
    return np.std(x) / np.sqrt(x.shape[0])


selected_models = [2, 16, 128]


## Model comparison
def jitter_plot(df: pd.DataFrame) -> p9.ggplot:
    n_symbols = len(df["symbol"].unique())
    metrics = df.groupby(
        ["tag", "symbol", "labels_per_batch"], as_index=False
    )["score"].agg(["mean", "std", stderr])
    df["tag"] = df.tag.map(lambda x: x.title())

    jitter = p9.position_jitterdodge(
        jitter_width=0.2, dodge_width=0.8, random_state=0
    )
    dodge = p9.position_dodge(width=0.8)
    return (
        p9.ggplot(df, p9.aes(x="symbol", color="tag"))
        + p9.geom_point(p9.aes(y="score"), position=jitter, size=1, alpha=0.4)
        + p9.geom_errorbar(
            p9.aes(
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
        + p9.labs(y="Similarity", x="Gene", color="Annotation")
        + p9.theme(
            axis_text_x=p9.element_text(
                rotation=45 if n_symbols > 10 else 0,
                ha="right" if n_symbols > 10 else "center",
                rotation_mode="anchor" if n_symbols > 10 else "default",
            ),
            legend_position="bottom",
            text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        )
    )


res = []
for model in 2 ** np.arange(1, 9):
    fpath = f"results/model_comparison/samples_abstract2gene_lpb_{model}.tsv"
    df = pd.read_table(fpath, index_col=0)
    df["labels_per_batch"] = model
    res.append(df)

df = pd.concat(res, axis=0, ignore_index=True)
df_avg = df.groupby(["pmid", "tag", "symbol"]).mean("score").reset_index()
df = df[np.isin(df.labels_per_batch, selected_models)]

p = jitter_plot(df)
p += p9.facet_wrap(
    "labels_per_batch",
    ncol=1,
    labeller=lambda val: f"{val} labels per batch",
)
p.save(
    f"figures/model_comparison/combined.{cfg.figure_ext}",
    width=cfg.fig_width,
    height=2 * cfg.fig_height,
)

p = jitter_plot(df_avg)
p.save(
    f"figures/model_comparison/ensemble.{cfg.figure_ext}",
    width=cfg.fig_width,
    height=cfg.fig_height,
)

## Behavioral studies
fpath = "results/predict_genes_in_behavioral_studies/correlations.tsv"
df = pd.read_table(fpath, index_col=0)
df = df[np.isin(df.model, selected_models)]

summary = (
    df.drop(columns=["parent"])
    .groupby(["model", "group"])
    .agg(["mean", "std"])
)
summary.columns = [col for _, col in summary.columns]
summary = summary.reset_index()

p = (
    p9.ggplot(df, p9.aes(x="correlation", fill="group", color="group"))
    + p9.geom_vline(
        p9.aes(xintercept="mean", color="group"),
        data=summary,
        linetype="dashed",
    )
    + p9.geom_histogram(alpha=0.3, binwidth=0.025, position="dodge")
    + p9.geom_text(
        p9.aes(label="mean", x="mean+0.035", y=90),
        color="black",
        size=8,
        va="bottom",
        data=summary,
        format_string="{:0.2f}",
    )
    + p9.facet_wrap(
        "model", ncol=1, labeller=lambda val: f"{val} labels per batch"
    )
    + p9.labs(fill="Type", color="Type", x="Correlation", y="Count")
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        legend_position="bottom",
    )
)
p.save(
    f"figures/predict_genes_in_behavioral_studies/combined.{cfg.figure_ext}",
    width=cfg.fig_width,
    height=2 * cfg.fig_height,
)


## Differential expression
def read_de(kind):
    fpath = f"results/differential_expression/events_{kind}.tsv"
    df = pd.read_table(fpath)
    df["kind"] = kind.title()
    return df


df = pd.concat([read_de(kind) for kind in ["molecular", "behavioral"]], axis=0)
df = df.drop(
    columns=[
        "events_AD",
        "events_other",
        "non_events_AD",
        "non_events_other",
        "relative_risk",
    ]
)

significant_genes = df.gene.unique()[
    df.groupby("gene").ci_low.agg(lambda x: any(x > 0))
]

df = df[np.isin(df.gene, significant_genes)]
df.loc[:, "alpha"] = df["ci_low"] > 0

categories = df.label.unique()
idx = categories != "PubTator3"
categories[idx] = sorted(categories[idx], key=lambda x: int(x.split(" ")[-1]))

cat_type = CategoricalDtype(categories=categories, ordered=True)
df["label"] = df["label"].astype(cat_type)

cat_type = CategoricalDtype(
    categories=["Molecular", "Behavioral"], ordered=True
)
df["kind"] = df["kind"].astype(cat_type)

dodge_col = p9.position_dodge(width=0.8)
p = (
    p9.ggplot(
        df,
        p9.aes(
            x="gene",
            y="ln_RR",
            color="label",
            alpha="alpha",
            linetype="alpha",
        ),
    )
    + p9.geom_hline(p9.aes(yintercept=0), color="black")
    + p9.geom_point(size=1, position=dodge_col)
    + p9.geom_errorbar(
        p9.aes(ymin="ci_low", ymax="ci_high"),
        width=0.3,
        size=0.3,
        position=dodge_col,
    )
    + p9.facet_wrap("kind", ncol=1)
    + p9.scale_alpha_discrete(range=(0.25, 1))
    + p9.scale_linetype_manual(values=("dashed", "solid"))
    + p9.scale_color_discrete()
    + p9.labs(
        y=r"$\log \left(\textrm{Relative Risk}\right)$",
        x="Gene",
        alpha=r"$\textrm{RR} > 1$\\(95\% confidence)",
        linetype=r"$\textrm{RR} > 1$\\(95\% confidence)",
        color="Annotations",
    )
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        axis_text_x=p9.element_text(rotation=45, ha="right"),
    )
)

p.save(
    f"figures/differential_expression/relative_risk.{cfg.figure_ext}",
    width=cfg.fig_width,
    height=2 * cfg.fig_height,
)
