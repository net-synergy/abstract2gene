from ._utils import default_cache_dir, default_data_dir
from .hgnc import download_gene_symbols

__all__ = ["default_cache_dir", "default_data_dir", "download"]


def download(content, data_dir=default_cache_dir()):
    """Download content from online DB

    If content already downloaded, checks for newer version and, if local files
    are outdated, downloads files. If local content is up-to-date, does
    nothing.

    Arguments
    ---------
    content : str { "gene_symbols" }, the name of the content to download.
    data_dir : str, where to download and check for content. Uses
        `default_cache_dir` by default.

    Returns
    -------
    None

    See also
    --------
    `abstract2gene.data.default_cache_dir`
    """

    downloaders = {"gene_symbols": download_gene_symbols}
    downloaders[content](data_dir)
