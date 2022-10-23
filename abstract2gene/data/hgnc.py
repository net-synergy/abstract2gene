import numpy as np
import pandas as pd

# from _utils import default_cache_dir


def gene_symbols():
    gene_file = get_gene_file()
    genes = pd.read_csv(gene_file, delimiter="\t", dtype=np.str_)
    return genes["symbol"].values


def get_gene_file():
    # TODO Should have method for downloading hgnc file into
    # `default_cache_dir` if not already downloaded then grab it, for now just
    # use manually downloaded file.

    return "~/data/hgnc_complete_set_2022-10-01.tsv"
