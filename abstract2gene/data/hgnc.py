import datetime
import os
import re
import warnings
from operator import itemgetter

import numpy as np
import pandas as pd
import requests

from ._utils import default_cache_dir

__all__ = ["gene_symbols", "download_gene_symbols"]

_base_url = (
    "http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/archive/monthly/tsv/"
)
_file_template = "hgnc_complete_set_{year}-{month}-01.txt"


def gene_symbols():
    """Return a list of gene symbols from HGNC."""

    gene_file = _find_gene_file()
    genes = pd.read_csv(gene_file, delimiter="\t", dtype=np.str_)
    return genes["symbol"].values


def _find_gene_file(data_dir=default_cache_dir()):
    cached_file = _most_recent_cached_file(data_dir)
    if not cached_file:
        raise FileNotFoundError(
            "Gene symbols have not been downloaded.\n\tDownload"
            ' with: abstract2gene.data.download("gene_symbols")'
        )

    today = datetime.date.today()
    latest_file = _file_template.format(year=today.year, month=today.month)
    if cached_file != latest_file:
        warnings.warn(
            "Local gene symbol file may be out of date.\n\tUpdate"
            ' with: abstract2gene.data.download("gene_symbols")'
        )

    return os.path.join(data_dir, cached_file)


def _most_recent_cached_file(data_dir):
    try:
        possible_gene_files = os.listdir(data_dir)
    except FileNotFoundError:
        return None

    pattern = _file_template.format(year=r"(\d{4})", month=r"(\d{2})")
    cached_file_dates = re.findall(pattern, "\0".join(possible_gene_files))
    if len(cached_file_dates) == 0:
        return None

    most_recent_file_date = sorted(cached_file_dates, key=itemgetter(0, 1))[-1]

    return _file_template.format(
        year=most_recent_file_date[0], month=most_recent_file_date[1]
    )


def download_gene_symbols(data_dir=default_cache_dir()):
    """Download HGNC's set of genes.

    If a file has already been downloaded, it will check if there's a newer
    version from the remote database and replace if there is.

    Arguments
    ---------
    data_dir : where to store the file (defaults to `default_cache_dir`)

    See also
    --------
    `abstract2gene.data.download` for a central download API for the package.
    `abstract2gene.data.default_cache_dir`
    """

    last_update = _latest_gene_symbol_update()
    latest_file = _file_template.format(
        year=last_update["year"], month=last_update["month"]
    )
    local_path = os.path.join(data_dir, latest_file)

    if os.path.exists(local_path):
        warnings.warn("Latest gene set already downloaded. Not doing anything")
        return

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    pattern = _file_template.format(year=r"\d{4}", month=r"\d{2}")
    stale_files = re.findall(pattern, "\0".join(os.listdir(data_dir)))

    if len(stale_files) > 0:
        print("Removing old gene symbol files...")

    for f in stale_files:
        os.remove(os.path.join(data_dir, f))

    print(f"Downloading {latest_file} to {data_dir}...")
    r = requests.get(os.path.join(_base_url, latest_file))
    with open(local_path, "wb") as f:
        f.write(r.content)

    return


def _latest_gene_symbol_update():
    r = requests.get(_base_url)
    pattern = _file_template.format(year=r"(\d{4})", month=r"(\d{2})")
    files = re.findall(pattern, r.text)
    return {"year": files[-1][0], "month": files[-1][1]}
