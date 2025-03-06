"""Defines data used throughout the examples.

These values can be changed here or in the examples themselves as needed. This
is intended purely for organization and reproducibility of results.
"""

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

MODELS = {
    # General Purpose models
    "ernie": "nghuyong/ernie-2.0-base-en",
    "mpnet": "microsoft/mpnet-base",
    "bert": "google-bert/bert-base-uncased",
    # Science fine-tuned models
    "specter": "sentence-transformers/allenai-specter",
    "specter2": "allenai/specter2_base",
    "scibert": "allenai/scibert_scivocab_uncased",
    "pubmedncl": "malteos/PubMedNCL",
}
