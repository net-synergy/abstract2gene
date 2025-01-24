"""Collect files from BioCXML as DataSets and Templates."""

__all__ = ["bioc2dataset"]

import concurrent.futures
import contextlib
import dataclasses
import functools
import os
import tarfile
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import datasets
import jax
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tqdm import tqdm

from abstract2gene.data import BiocDownloader, PubmedDownloader


def bioc2dataset(
    archives: Iterable[int],
    n_files: int = -1,
    ann_type="Gene",
    batch_size: int = 1000,
    max_cpu: int = 1,
    mask_token: str = "[MASK]",
    cache_dir: str | None = None,
) -> datasets.Dataset:
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
    ann_type : str, default "Gene"
        Which annotation type to mask and use for labels. Values should be
        exactly as presented in the XML files. Only annotations for abstract
        passages are used not those from titles are full text. Annotations for
        the selected annotation type are stripped from the abstracts before
        embedding.
    batch_size : int, 1000
        Batch size used for dataset map functions.
    max_cpu : int, default 1
        How many process to run in parallel for CPU.
    mask_token : str, default "[MASK]"
        String to mask words with with. Should be specific to the tokenizer
        being used.
    cache_dir : Optional str
        Where to cache and retrieve BioC archives locally. Defaults to
        `abstract2gene.storage.default_cache_dir`. A different cache can also
        be set globally using `abstract2gene.storage.set_cache_dir`.

    Returns
    -------
    dataset : datasets.Dataset
        A dataset containing the masked abstract for each publication and a
        list of labels.

    """

    def read_pubmed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: Assumes genes even if another annotation type is requested.
        pubmed_downloader = PubmedDownloader()
        gene2pubmed, gene_info = pubmed_downloader.download()
        links = pd.read_table(
            gene2pubmed,
            header=0,
            names=["TaxID", "GeneID", "PMID"],
            usecols=["PMID", "GeneID"],
        ).sort_values("PMID")

        info = pd.read_table(
            gene_info, usecols=["GeneID", "Symbol"]
        ).sort_values("GeneID")

        return (links, info)

    def attach_pubmed_genes(
        examples: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        def get_genes(pmid: int) -> list[int]:
            start = np.searchsorted(pubmed_edges["PMID"], pmid)
            end = start
            while (end < pubmed_edges.shape[0]) and (
                pubmed_edges["PMID"].iloc[end] == pmid
            ):
                end += 1

            return list(pubmed_edges["GeneID"].iloc[start:end])

        return {"gene2pubmed": [get_genes(pmid) for pmid in examples["pmid"]]}

    def deduplicate_symbols(symbol_table: pd.DataFrame) -> list[str]:
        dups, counts = np.unique(symbol_table["Symbol"], return_counts=True)
        dups = dups[counts > 1]
        dup_counter = dict(zip(dups, np.zeros((dups.shape[0],), dtype=int)))
        dup_mask = np.isin(symbol_table["Symbol"], dups)
        symbols = symbol_table["Symbol"].array.tolist()
        for i in np.arange(dup_mask.shape[0])[dup_mask]:
            dup_counter[symbols[i]] += 1
            symbols[i] = f"{symbols[i]}_{dup_counter[symbols[i]]}"

        return symbols

    def repack_labels(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Shrink label IDs to a sequential range."""
        return {
            k: [
                [id2idx[gene_id] for gene_id in pub_genes]
                for pub_genes in examples[k]
            ]
            for k in ("gene2pubmed", "gene2pubtator")
        }

    tar_paths = _get_archives(archives, cache_dir)
    args = ParsingArgs(ann_type, mask_token, n_files)

    print(f"Reading {len(tar_paths)} archives...")
    dataset = datasets.Dataset.from_list(
        _grab_publication_data(tar_paths, args, max_cpu)
    )

    # If first batch of pubmed genes is all empty (unlikely with a large
    # batch), datasets will detect a null feature type instead of using
    # int. Explicitly pass it the correct type to prevent this.
    features = dataset.features.copy()
    features["gene2pubmed"] = features["gene2pubtator"]

    pubmed_edges, info = read_pubmed_data()
    dataset = dataset.map(
        attach_pubmed_genes,
        batched=True,
        batch_size=batch_size,
        features=features,
        num_proc=max_cpu,
        desc="Attach pubmed genes",
    )

    gene_ids: ArrayLike = jax.tree.leaves(
        [dataset["gene2pubmed"], dataset["gene2pubtator"]]
    )
    gene_ids = np.unique_values(gene_ids)

    symbol_table = info.merge(
        pd.DataFrame({"GeneID": gene_ids}), how="right", on="GeneID"
    )
    symbol_table["Symbol"] = symbol_table["Symbol"].fillna(
        symbol_table["GeneID"].transform(str)
    )

    symbols = deduplicate_symbols(symbol_table)
    id2idx = dict(zip(symbol_table["GeneID"], symbol_table.index))

    features["gene2pubtator"] = datasets.Sequence(
        datasets.ClassLabel(names=symbols)
    )
    features["gene2pubtator"].feature.gene_ids = gene_ids
    features["gene2pubmed"] = features["gene2pubtator"]

    return dataset.map(
        repack_labels,
        batched=True,
        batch_size=batch_size,
        features=features,
        num_proc=max_cpu,
        desc="Repack label IDs",
    )


@dataclasses.dataclass
class ParsingArgs:
    ann_type: str
    mask_token: str
    n_files: int


def _get_archives(
    file_numbers: Iterable[int],
    cache_dir: str | None,
) -> list[str]:
    """Find the local path to the BioC archives."""
    downloader = BiocDownloader(cache_dir=cache_dir)
    downloader.file_numbers = file_numbers
    return downloader.download()


def _grab_publication_data(
    archives: list[str], args: ParsingArgs, max_cpu
) -> list[dict[str, Any]]:
    parser = functools.partial(_parse_tarfile, args=args)

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

    return [publication for file in pub_data for publication in file]


def _parse_tarfile(
    archive: str,
    run_id: int,
    total_runs: int,
    position: int,
    args: ParsingArgs,
) -> list[dict[str, Any]]:
    with tarfile.open(archive, "r") as tar:
        n_files = args.n_files if args.n_files > 0 else len(tar.getnames())
        run_files = [
            tar.getnames()[i] for i in range(run_id, n_files, total_runs)
        ]
        parser = functools.partial(
            _parse_file,
            tar=tar,
            ann_type=args.ann_type,
            mask_token=args.mask_token,
        )
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
    member: str, tar: tarfile.TarFile, ann_type: str, mask_token: str
) -> list[dict[str, Any] | None]:
    with tar.extractfile(member) as fd:  # type: ignore[union-attr]
        try:
            tree = ET.parse(fd)
        except ET.ParseError as e:
            print(f"Failed to parse {member}:\n  {e.msg}")
            return [None]

        parser = functools.partial(
            _parse_doc, ann_type=ann_type, mask_token=mask_token
        )
        return [parser(doc) for doc in tree.iterfind("document")]


def _parse_doc(doc, ann_type: str, mask_token: str) -> dict[str, Any] | None:
    try:
        id_field = doc.find("id")

        # I don't think this should ever happen so raising an error to make it
        # obvious if this scenario occurs.
        if id_field is None:
            raise ValueError("Document missing ID field.")

        pmid = id_field.text
        if pmid is None:
            raise ValueError("Document missing ID field.")

        passage = doc.find("passage")
        if pmid.startswith("PMC") and passage:
            pmids = [
                infon.text
                for infon in passage.iterfind("infon")
                if infon.attrib["key"] == "article-id_pmid" and infon.text
            ]
            if len(pmids):
                pmid = pmids[0]
            else:
                raise ValueError("Document missing ID field.")

        year: int | None = None
        if passage:
            years = [
                infon.text
                for infon in passage.iterfind("infon")
                if infon.attrib["key"] == "year" and infon.text
            ]
            year = int(years[0]) if len(years) else None

        abstracts_elements = _get_abstracts(doc)
        masked_abstract = "".join(
            _mask_abstract(abstract, ann_type, mask_token)
            for abstract in abstracts_elements
        )

        annotations = [
            ann_id
            for abstract in abstracts_elements
            for ann_id in _collect_annotations(abstract, ann_type)
            if ann_id is not None
        ]

        return {
            "pmid": int(pmid),
            "abstract": masked_abstract,
            "year": year,
            "gene2pubtator": list(set(annotations)),
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


def _mask_abstract(passage: ET.Element, ann_type: str, mask_token: str) -> str:
    offset_field = passage.find("offset")
    if offset_field is None:
        raise ValueError("Abstract missing offset field")

    offset = int(offset_field.text) if offset_field.text else 0
    text_field = passage.find("text")
    if text_field is None:
        raise ValueError("Abstract missing text field")
    text = text_field.text if text_field.text else ""

    mask = np.ones((len(text),))
    for ann in passage.iterfind("annotation"):
        if not _is_type(ann, ann_type):
            continue

        loc = ann.find("location")
        if loc is None:
            continue

        pos_start = int(loc.attrib["offset"]) - offset
        pos_end = pos_start + int(loc.attrib["length"])
        mask[pos_start + 1 : pos_end] = 0
        mask[pos_start] = -1

    return "".join(
        (
            letter if mask[i] > 0 else mask_token
            for i, letter in enumerate(text)
            if mask[i]
        )
    )


def _is_type(annotation: ET.Element, ann_type: str) -> bool:
    return any(
        (
            (infon.attrib["key"] == "type") and (infon.text == ann_type)
            for infon in annotation.iterfind("infon")
        )
    )


def _collect_annotations(
    passage: ET.Element, ann_type: str
) -> list[int | None]:
    return [
        _get_identifier(ann)
        for ann in passage.iterfind("annotation")
        if _is_type(ann, ann_type)
    ]


def _get_identifier(annotation: ET.Element) -> int | None:
    for infon in annotation.iterfind("infon"):
        if infon.attrib["key"] == "identifier":
            ann_id: int | None = None
            if infon is not None:
                # ValueError can occur when multiple IDs are given for a
                # single gene, i.e. int("112;3939"). I don't know why a
                # gene should have multiple IDs and it is a rare occurrence
                # so these cases are dropped instead of picking an ID at
                # random.
                with contextlib.suppress(ValueError):
                    ann_id = (
                        int(infon.text) if infon.text is not None else None
                    )

            return ann_id

    return None
