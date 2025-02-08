"""Collect files from BioCXML files to a huggingface dataset."""

__all__ = ["bioc2dataset"]

import concurrent.futures
import functools
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Iterable

import datasets
import jax
import numpy as np
from tqdm import tqdm

from abstract2gene.data import BiocDownloader

# Skip SNPs and mutations as they're hard to work with and rare. Could come
# back to them later.
ANN_TYPES = (
    "gene",
    "disease",
    "species",
    "chemical",
    "cellline",
    "chromosome",
)


def bioc2dataset(
    archives: Iterable[int],
    n_files: int = -1,
    batch_size: int = 1000,
    max_cpu: int = 1,
    cache_dir: str | None = None,
) -> datasets.DatasetDict:
    """Create a dataset from Pubtator's BioC archive files.

    Pubtator's archive files contain abstracts plus annotations. Using these
    files instead of the pubtator gene files allows getting abstract text from
    BioC and masking the selecting annotation type from the abstracts before
    embedding the abstracts to ensure the models aren't learning specific
    symbols used for prediction.

    Parameters
    ----------
    archives : list[int]
        Which archives to use. There are 10 archives, each contains files
        ending with a given digit (archive '0' contains all numbered files
        where zero is the last digit). By default parses all files contained in
        each provided archive. (Note: archives are ~16GB each to download and
        have ~1e6 publications each).
    n_files : int, optional
        The number of files to parse per archive. By default, parses all files.
        This is most sensible when using only a single archive to create a
        small dataset. Each file contains 100 abstracts and there is about 1e4
        files per archive.
    batch_size : int, 1000
        Batch size used for dataset map functions.
    max_cpu : int, default 1
        How many process to run in parallel for CPU.
    cache_dir : Optional str
        Where to cache and retrieve BioC archives locally. Defaults to
        `abstract2gene.storage.default_cache_dir`. A different cache can also
        be set globally using `abstract2gene.storage.set_cache_dir`.

    Returns
    -------
    dataset : datasets.DatasetDict
        A dataset containing the masked abstract for each publication and a
        list of labels. The dataset is split based on the archives the data
        originated from.

    Examples
    --------
    To get all data as into a single dataset instead of multiple splits:

    >> from datasets import concatenate_datasets
    >> dataset_dict = bioc2dataset(list(range(10)))
    >> dataset = concatenate_datasets([ds for ds in dataset_dict.values()])

    """

    def repack_labels(
        examples: dict[str, list[Any]],
        id2idx: dict[str, dict[str, int]],
    ) -> dict[str, list[Any]]:
        """Shrink label IDs to a sequential range."""
        return {
            k: [
                [int(id2idx[k][ann_id]) for ann_id in pub_anns]
                for pub_anns in examples[k]
            ]
            for k in ANN_TYPES
        }

    tar_paths = _get_archives(archives, cache_dir)
    print(f"Reading {len(tar_paths)} archives...")
    dataset = datasets.DatasetDict(
        {
            archive: datasets.Dataset.from_list(data)
            for archive, data in _grab_publication_data(
                tar_paths, n_files, max_cpu
            ).items()
        }
    )

    key = list(dataset.keys())[0]
    features = dataset[key].features.copy()
    id2idx: dict[str, dict[str, int]] = {}
    for ann_type in ANN_TYPES:
        ann_ids = jax.tree.leaves([ds[ann_type] for ds in dataset.values()])
        ann_ids = np.unique_values(ann_ids).tolist()
        id2idx[ann_type] = dict(zip(ann_ids, range(len(ann_ids))))
        features[ann_type] = datasets.Sequence(
            datasets.ClassLabel(names=ann_ids)
        )

    dataset = dataset.map(
        repack_labels,
        batched=True,
        batch_size=batch_size,
        features=features,
        num_proc=max_cpu,
        fn_kwargs={"id2idx": id2idx},
        desc="Repack label IDs",
    )

    return datasets.DatasetDict(dataset)


def _get_archives(
    file_numbers: Iterable[int],
    cache_dir: str | None,
) -> list[str]:
    """Find the local path to the BioC archives."""
    downloader = BiocDownloader(cache_dir=cache_dir)
    downloader.file_numbers = file_numbers
    return downloader.download()


def _grab_publication_data(
    archives: list[str], n_files: int, max_cpu
) -> dict[str, list[dict[str, Any]]]:
    parser = functools.partial(_parse_tarfile, n_files=n_files)

    n_iters = max(max_cpu, len(archives))
    archive_reps = [archives[i % len(archives)] for i in range(n_iters)]
    runs = [i // len(archives) for i in range(n_iters)]
    total_runs = [
        max((runs[i] + 1 for i in range(idx, n_iters, len(archives))))
        for idx in range(len(archives))
    ]
    total_runs = [total_runs[i % len(archives)] for i in range(len(runs))]
    position = list(range(len(total_runs)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpu) as exe:
        pub_data = list(
            exe.map(parser, archive_reps, runs, total_runs, position)
        )

    def simplify_name(name: str) -> str:
        m = re.search(r"BioCXML\.(\d)\.tar\.gz", name)
        if m is None:
            raise ValueError("Got unexpected archive name")

        return f"BioCXML_{m.group(1)}"

    out = defaultdict(list)
    for archive, data in zip(archive_reps, pub_data):
        out[simplify_name(archive)].append(data)

    return {k: [el for ls in parts for el in ls] for k, parts in out.items()}


def _parse_tarfile(
    archive: str,
    run_id: int,
    total_runs: int,
    position: int,
    n_files: int,
) -> list[dict[str, Any]]:
    with tarfile.open(archive, "r") as tar:
        max_files = len(tar.getnames())
        n_files = min(max_files, n_files) if n_files > 0 else max_files
        run_files = [
            tar.getnames()[i] for i in range(run_id, n_files, total_runs)
        ]
        parser = functools.partial(_parse_file, tar=tar)
        desc = os.path.basename(archive) + f" ({run_id + 1}/{total_runs})"
        return [
            pubs
            for file_data in (
                parser(f)
                for f in tqdm(run_files, desc=desc, position=position)
            )
            for pubs in file_data
            if pubs is not None
        ]


def _parse_file(
    member: str, tar: tarfile.TarFile
) -> list[dict[str, Any] | None]:
    with tar.extractfile(member) as fd:  # type: ignore[union-attr]
        try:
            tree = ET.parse(fd)
        except ET.ParseError as e:
            print(f"Failed to parse {member}:\n  {e.msg}")
            return [None]

        return [_parse_doc(doc) for doc in tree.iterfind("document")]


def _parse_doc(doc) -> dict[str, Any] | None:
    try:
        id_field = doc.find("id")

        # I don't think this should ever happen so raising an error to make it
        # obvious if this scenario occurs.
        if id_field is None:
            raise ValueError("Document missing ID field.")

        pmid = id_field.text
        if pmid is None:
            raise ValueError("Document missing ID field.")

        title_passage = doc.find("passage")
        if pmid.startswith("PMC") and title_passage:
            pmid = _collect_infon(title_passage, "article-id_pmid")

            if pmid is None:
                raise ValueError("Document missing ID field.")

        year: int | None = None
        if title_passage:
            year_str = _collect_infon(title_passage, "year")
            year = int(year_str) if year_str else None

            title = (
                title_passage.find("text").text
                if title_passage.find("text")
                else ""
            )

        abstract_elements = _get_abstracts(doc)

        def collect_text(element) -> str:
            text_field = element.find("text")
            if text_field is None:
                raise ValueError("Abstract missing text field")

            return text_field.text if text_field.text else ""

        def get_offset(element) -> int:
            offset = element.find("offset")
            if offset is None:
                raise ValueError("Missing offset")

            return int(offset.text)

        abs_text = [collect_text(abstract) for abstract in abstract_elements]
        abstract = " ".join(abs_text)

        offsets = [get_offset(abstract) for abstract in abstract_elements]
        for i in range(1, len(offsets)):
            offsets[i] -= sum(len(text) + 1 for text in abs_text[:i])

        annotations = [
            annotation
            for offset, abstract in zip(offsets, abstract_elements)
            for annotation in _collect_annotations(abstract, offset)
        ]

        ann_data: dict[str, set[str]] = {k: set() for k in ANN_TYPES}
        for identifier, meta in annotations:
            if meta["type"].lower() in ann_data:
                ann_data[meta["type"].lower()].add(identifier)

        _, meta_data = zip(*annotations)

        return {
            **{
                "pmid": int(pmid),
                "year": year,
                "title": title,
                "abstract": abstract,
                "annotation": list(meta_data),
            },
            **{k: list(v) for k, v in ann_data.items()},
        }
    except ValueError:
        return None


def _get_abstracts(doc: ET.Element) -> list[ET.Element]:
    return [
        passage
        for passage in doc.iterfind("passage")
        if any(
            (
                (infon.attrib["key"] == "type" and infon.text == "abstract")
                for infon in passage.iterfind("infon")
            )
        )
    ]


def _collect_infon(passage: ET.Element, key: str) -> str | None:
    """Return first item that matches key."""
    for infon in passage.iterfind("infon"):
        if infon.attrib["key"] == key:
            return infon.text

    return None


def _collect_annotations(
    passage: ET.Element, offset: int
) -> list[tuple[str, dict]]:
    return [
        (ann_id, meta)
        for (ann_id, meta) in (
            _read_annotation(annotation, offset)
            for annotation in passage.iterfind("annotation")
        )
        if (ann_id is not None) and (meta is not None)
    ]


def _read_annotation(
    annotation: ET.Element, offset: int
) -> tuple[str | None, dict | None]:
    ann_id: str | None = None
    ann_type: str | None = None
    for infon in annotation.iterfind("infon"):
        if infon.attrib["key"] == "identifier" and infon is not None:
            ann_id = infon.text if infon.text is not None else None
        if infon.attrib["key"] == "type" and infon is not None:
            ann_type = infon.text if infon.text is not None else None

    location = annotation.find("location")
    if (
        (ann_id is None)
        or (ann_type is None)
        or (location is None)
        or (ann_id == "-")
    ):
        return (None, None)

    offset = int(location.attrib["offset"]) - offset
    length = int(location.attrib["length"])

    return (ann_id, {"type": ann_type, "offset": offset, "length": length})
