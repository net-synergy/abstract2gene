import os

import numpy as np
import pandas as pd
import plotnine as p9
from pandas.api.types import CategoricalDtype
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

import example._config as cfg

EXPERIMENT = "fine_tune_experiments"
FIGDIR = f"figures/{EXPERIMENT}"

if not os.path.exists(FIGDIR):
    os.mkdir(FIGDIR)


def extract_metric(metric, eval_type, model, experiment):
    event_acc = EventAccumulator(f"logs/{model}/{experiment}")
    event_acc.Reload()

    time, steps, value = zip(
        *[
            (event.wall_time, event.step, event.value)
            for event in event_acc.Scalars(metric)
        ]
    )

    # In case there are multiple training steps. Need to sort results in time.
    df = pd.DataFrame(
        {
            "time": time,
            "step": steps,
            "value": value,
            "model": model,
            "experiment": experiment,
            "eval_type": eval_type,
        }
    ).sort_values("time")
    delta = df["step"][1] - df["step"][0]
    df["step"] = list(range(delta, delta * (df.shape[0] + 1), delta))
    return df[np.logical_not(df["experiment"] == "multi_phase")]


def _locate(df, step, experiment, eval_type, pos, label):
    df_new = df.loc[
        np.all(
            np.vstack(
                [
                    df["step"] == step,
                    df["experiment"] == experiment,
                    df["eval_type"] == eval_type,
                ]
            ).T,
            axis=1,
        ),
        ["step", "model", "value"],
    ]
    df_new[pos] = df_new["value"]
    df_new["label"] = label

    return df_new


def _standardize_names(column: pd.Series) -> pd.Series:
    return column.apply(lambda x: x.replace("_", " ").title())


def _preprocess(df):
    df.loc[:, "experiment"] = _standardize_names(df["experiment"])
    df.loc[:, "eval_type"] = _standardize_names(df["eval_type"])

    if len(df.experiment.unique()) > 1:
        unmasked = df[
            np.logical_and(
                df.experiment == "Unmasked", df.eval_type == "Unmasked"
            )
        ]
        x_lim = min(
            [
                unmasked.loc[unmasked["model"] == model, "step"].max()
                for model in unmasked.model.unique()
            ]
        )
        df = df[df["step"] <= x_lim]

    experiments = df.sort_values("time").experiment.unique()
    cat_type = CategoricalDtype(categories=experiments, ordered=True)
    df.loc[:, "experiment"] = df["experiment"].astype(cat_type)

    return df


def plot_training_strategies(df):
    df = _preprocess(df)

    gene_and_disease_name = r"\noindent{}Genes And\\Disease Masked"
    df.loc[df["eval_type"] == "Genes And Disease Masked", "eval_type"] = (
        gene_and_disease_name
    )

    eval_types = ["Unmasked", "Genes Masked", gene_and_disease_name]
    cat_type = CategoricalDtype(categories=eval_types, ordered=True)
    df["eval_type"] = df["eval_type"].astype(cat_type)

    diff_positions = {
        "step": [6750, 7500, 8250, 9000, 9750],
        "experiment": [
            {"top": "Gene Only", "bottom": "Gene Only"},
            {"top": "Unmasked", "bottom": "Gene Only"},
            {"top": "Gene Only", "bottom": "Gene Only"},
            {"top": "Gene Only", "bottom": "Gene And Disease"},
            {"top": "Permute", "bottom": "Gene Only"},
        ],
        "eval_type": [
            {"top": "Unmasked", "bottom": "Genes Masked"},
            {"top": "Unmasked", "bottom": "Genes Masked"},
            {"top": "Genes Masked", "bottom": gene_and_disease_name},
            {"top": "Genes Masked", "bottom": gene_and_disease_name},
            {"top": "Unmasked", "bottom": "Unmasked"},
        ],
        "label": ["1", "2", "3", "4", "5"],
    }

    diffs = pd.concat(
        [
            pd.merge(
                *[
                    _locate(df, step, exp[pos], eval_type[pos], pos, label)
                    for pos in ["top", "bottom"]
                ],
                how="inner",
                on=["label", "model", "step"],
            )
            for step, exp, eval_type, label in zip(
                diff_positions["step"],
                diff_positions["experiment"],
                diff_positions["eval_type"],
                diff_positions["label"],
            )
        ]
    )

    return (
        p9.ggplot(
            df,
            p9.aes(
                x="step",
                y="value",
                color="experiment",
                shape="eval_type",
                linetype="eval_type",
            ),
        )
        + p9.geom_smooth(method="loess", se=False, size=0.5)
        + p9.geom_point(size=1)
        + p9.geom_point(
            p9.aes(alpha="eval_type"),
            size=0.75,
            color="white",
        )
        + p9.geom_linerange(
            p9.aes(x="step", ymin="bottom", ymax="top"),
            data=diffs,
            inherit_aes=False,
        )
        + p9.geom_label(
            p9.aes(
                x="step - 280",
                y="(bottom + top) / 2",
                label="label",
            ),
            size=8,
            data=diffs,
            inherit_aes=False,
        )
        + p9.scale_alpha_discrete((0, 0.8))
        + p9.scale_y_continuous(minor_breaks=4, limits=(0.8, 0.95))
        + p9.scale_color_discrete()
        + p9.labs(
            x="Step",
            y="Accuracy",
            color="Experiment",
            fill="Experiment",
            shape="Evaluation",
            linetype="Evaluation",
            alpha="Evaluation",
        )
        + p9.theme(
            text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        )
        + p9.facet_wrap("model", ncol=1)
    )


def plot_all_models(df):
    df = df[df["experiment"] == "permute"]
    df = df[df["eval_type"] != "genes_masked"]
    df = _preprocess(df)

    eval_types = sorted(df.eval_type.unique(), reverse=True)
    cat_type = CategoricalDtype(categories=eval_types, ordered=True)
    df["eval_type"] = df["eval_type"].astype(cat_type)

    return (
        p9.ggplot(df, p9.aes(x="step", y="value", color="model"))
        + p9.geom_smooth(method="loess", se=False, size=0.5)
        + p9.geom_point(size=1)
        + p9.scale_color_discrete()
        + p9.labs(x="Step", y="Accuracy", color="Model")
        + p9.theme(
            text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
        )
        + p9.facet_wrap("eval_type", ncol=1)
    )


logs = os.listdir("logs")
models = [
    model
    for model in (model.split("/")[-1] for model in logs)
    if not model.startswith("_")
]
accuracy = pd.concat(
    [
        extract_metric(
            f"eval/{eval_type}_cosine_accuracy", eval_type, model, exp
        )
        for model in models
        for exp in os.listdir(os.path.join("logs", model))
        for eval_type in [
            "unmasked",
            "genes_masked",
            "genes_and_disease_masked",
        ]
    ],
    axis=0,
    ignore_index=True,
)

p = plot_training_strategies(
    accuracy[np.isin(accuracy.model, ("MPNet", "PubMedNCL"))]
)
p.save(
    os.path.join(FIGDIR, f"training_strategies.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height * 2,
)

p = plot_all_models(accuracy)
p.save(
    os.path.join(FIGDIR, f"training_curves.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height * 2,
)
