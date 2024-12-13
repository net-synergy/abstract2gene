"""Download relevant data from various sources."""

from .bioc import BiocDownloader
from .pubmed import PubmedDownloader
from .pubtator import PubtatorDownloader

__all__ = ["download"]


def download(content: str, cache_dir: str | None = None) -> None:
    """Download content from online FTP server.

    If content already downloaded, checks for newer version and, if local files
    are outdated, downloads files. If local content is up-to-date, does
    nothing.

    Parameters
    ----------
    content : str { "pubmed", "pubtator", "bioc" }
        The name of the content to download.
    cache_dir : str, optional
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
        "pubmed": PubmedDownloader,
        "pubtator": PubtatorDownloader,
        "bioc": BiocDownloader,
    }
    downloader = downloaders[content](cache_dir)
    downloader.download()
