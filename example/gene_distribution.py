import os

import datasets
import jax
import numpy as np
import pandas as pd
import plotnine as p9

import example._config as cfg
from abstract2gene.dataset import mutators

FIGDIR = "figures/gene_distribution"

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

dataset = datasets.load_dataset(f"{cfg.hf_user}/pubtator3_abstracts")["train"]
dataset = mutators.translate_to_human_orthologs(dataset, cfg.max_cpu)
counts = np.bincount(jax.tree.leaves(dataset["gene"]))

df = pd.DataFrame({"Occurrences": counts})
p = (
    p9.ggplot(df)
    + p9.geom_histogram(p9.aes(x="Occurrences", y=p9.after_stat("count")))
    + p9.labs(y="Count")
    + p9.scale_x_log10()
    + p9.theme(
        text=p9.element_text(family=cfg.font_family, size=cfg.font_size),
    )
)

p.save(
    os.path.join(FIGDIR, f"histogram.{cfg.figure_ext}"),
    width=cfg.fig_width,
    height=cfg.fig_height,
)
