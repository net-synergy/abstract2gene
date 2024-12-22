"""Download relevant data from various sources."""

__all__ = ["download", "default_cache_dir", "default_data_dir"]

from typing import Iterable

from abstract2gene.storage import default_cache_dir, default_data_dir

from .bioc import BiocDownloader
from .pubmed import PubmedDownloader
from .pubtator import PubtatorDownloader


def download(
    content: str,
    files: Iterable[str] | None = None,
    cache_dir: str | None = None,
    check_remote: bool = False,
) -> list[str]:
    """Download content from online FTP server.

    If content already downloaded, checks for newer version and, if local files
    are outdated, downloads files. If local content is up-to-date, does
    nothing.

    Parameters
    ----------
    content : str { "pubmed", "pubtator", "bioc" }
        The name of the content to download.
    files : list[str], optional
        If files is given, download these files from the FTP server instead of
        the default file list.
    cache_dir : str, optional
        Where to download and check for content. Uses `default_cache_dir` by
        default.
    check_remote : bool, default True
        Whether to connect to the remote server and download files. If True,
        this checks for cached files, downloads missing files and cached files
        that are older than server's. If False,returns the cached files without
        checking for updates from the server. An error is raised if any file is
        missing from the cache. This must be set to True the first time this
        command is run to cache the files initially. When set to False,

    Returns
    -------
    files : list of file paths.

    See Also
    --------
    `abstract2gene.data.default_cache_dir`

    """
    downloaders = {
        "pubmed": PubmedDownloader,
        "pubtator": PubtatorDownloader,
        "bioc": BiocDownloader,
    }
    downloader = downloaders[content](
        cache_dir=cache_dir, check_remote=check_remote
    )

    return downloader.download(files=files)
