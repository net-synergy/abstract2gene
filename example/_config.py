"""Defines data used throughout the examples.

These values should be changed in an `a2g.toml` file at the root of the
project. This is intended purely for organization and reproducibility of
results. Some configuration options (such as data used for different
experiments) cannot be set in the toml due to the awkwardness of expressing
them combine with it not likely being necessary. If you want to change those
anyway, change directly in this file.

Values that are not settable in the toml file are styled in uppercase while the
variables that are dependent on the value in the toml file are in lowercase.
"""

import tomllib
from math import sqrt

import matplotlib.pyplot as plt

_conf_file = "a2g.toml"
try:
    with open(_conf_file, "rb") as fp:
        conf = tomllib.load(fp)["experiments"]
except (KeyError, FileNotFoundError):
    conf = {}

_file_template = "data/BioCXML_{archive}/data-{f_idx:05}-of-{f_total:05}.arrow"

EMBEDDING_TRAIN_FILES = [
    _file_template.format(archive=9, f_idx=i, f_total=21) for i in range(10)
]
A2G_TRAIN_FILES = [
    _file_template.format(archive=9, f_idx=i, f_total=21)
    for i in range(10, 15)
]
LABEL_SIMILARITY_FILES = [
    _file_template.format(archive=8, f_idx=i, f_total=21) for i in range(2)
]
AD_DE_FILES = [
    _file_template.format(archive=8, f_idx=i, f_total=21) for i in range(2, 15)
]
TEST_FILES = [_file_template.format(archive=7, f_idx=0, f_total=21)]

# Default models, always available through the cfg.MODELS but only models
# in cfg.models will be fine-tuned. If "embedding-models" set in the TOML those
# will be used for models instead of MODELS.
MODELS = {
    # General Purpose models
    "ERNIE": "nghuyong/ernie-2.0-base-en",
    "MPNet": "microsoft/mpnet-base",
    "BERT": "google-bert/bert-base-uncased",
    # Science fine-tuned models
    "SPECTER": "sentence-transformers/allenai-specter",
    "SPECTER2": "allenai/specter2_base",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "PubMedNCL": "malteos/PubMedNCL",
}

hf_user = "dconnell"
figure_type = "png"
max_cpu = 1
template_size = 32

# To ensure reproducible results, each script is passed a random seed to use.
# The random seed is a function of the script run order (the ith script gets 10
# * i as a seed). By using multiples of 10 each script can use up to 10 seeds
# if it needs more than one. Scripts don't necessarily use a seed but always
# get one.
seeds = {
    "create_from_bioc": 10,
    "embedding_model_selection": 20,
    "finetune_encoder": 30,
    "train_abstract2gene": 40,
    "test_abstract2gene": 50,
    "label_embedding_similarity": 60,
    "reference_similarity": 70,
    "predict_genes_in_behavioral_studies": 80,
    "differential_expression": 90,
}

hf_user = conf.get("hf_user", hf_user)
models = conf.get("embedding-models", MODELS)
max_cpu = conf.get("max_cpu", max_cpu)
template_size = conf.get("template_size", template_size)
text_width = 7.5
font_family: str | None = None
font_size: int | None = None

if "seeds" in conf:
    seeds.update(conf["seeds"])

if "figures" in conf:
    _fig_conf = conf["figures"]
    figure_ext = _fig_conf.get("type", figure_type)
    "type" in _fig_conf and _fig_conf.pop("type")
    text_width = _fig_conf.get("text_width", text_width)
    "text_width" in _fig_conf and _fig_conf.pop("text_width")

    _rename = {
        "use_tex": "text.usetex",
        "font_family": "font.family",
        "font_size": "font.size",
        "dpi": "dpi",
    }
    plt.rcParams.update({_rename[k]: v for k, v in _fig_conf.items()})

    font_family = _fig_conf.get("font_family", font_family)
    font_size = _fig_conf.get("font_size", font_size)

fig_width = text_width * 0.9
fig_height = fig_width * ((0.5 * (1 + sqrt(5))) - 1)
