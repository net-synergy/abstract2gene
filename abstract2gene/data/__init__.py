"""Download relevant publication data from various sources."""

from ._utils import default_cache_dir
from .hgnc import download_gene_symbols as download_hgnc_gene_symbols
from .pubtator import download_gene_edges as download_gene_edges

__all__ = ["default_cache_dir", "download"]


def download(content, cache_dir=None):
    """Download content from online DB.

    If content already downloaded, checks for newer version and, if local files
    are outdated, downloads files. If local content is up-to-date, does
    nothing.

    Parameters
    ----------
    content : str { "hgnc_genes", "pubtator_genes", "pubmed_genes" }
        The name of the content to download.
    cache_dir : optional str
        Where to download and check for content. Uses `default_cache_dir` by
        default.

    Returns
    -------
    None

    See Also
    --------
    `abstract2gene.data.default_cache_dir`

    """
    downloaders = {
        "hgnc_genes": download_hgnc_gene_symbols,
        "pubmed_genes": lambda _dir: download_gene_edges("pubmed", _dir),
        "pubtator_genes": lambda _dir: download_gene_edges("pubtator", _dir),
    }
    downloaders[content](cache_dir or default_cache_dir())
