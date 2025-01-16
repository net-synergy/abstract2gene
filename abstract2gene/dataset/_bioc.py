"""Collect files from BioCXML as DataSets and Templates."""

__all__ = ["bioc2dataset"]

import contextlib
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterable, Sequence

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from abstract2gene.data import BiocDownloader, PubmedDownloader


def bioc2dataset(
    archives: Iterable[int],
    n_files: int = -1,
    ann_type="Gene",
    batch_size: int = 32,
    max_tokens: int = 512,
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
    batch_size : int, default 50
        Batch size used for embedding. Larger batch sizes may result in faster
        processing but can exhaust memory. 50 was chosen as default since it
        divides the number of publications in a file and the model is passed
        all abstracts from a file at once. Note this is unrelated to the
        resulting dataset's batch size.
    max_tokens : int, default 512
        The number of tokens the tokenizer will return. The tokenizer creates
        arrays of tokens with exactly this many tokens, truncated longer
        abstracts and padding smaller abstracts. This allows the model to run
        on multiple abstracts at once. Larger values take longer to process but
        smaller values risk dropping information.
    cache_dir : Optional str
        Where to cache and retrieve BioC archives locally. Defaults to
        `abstract2gene.storage.default_cache_dir`. A different cache can also
        be set globally using `abstract2gene.storage.set_cache_dir`.

    Returns
    -------
    dataset : datasets.Dataset
        A dataset containing the abstract embeddings for each publication and a
        list of labels.

    """
    parser = _BiocParser(
        archives, n_files, ann_type, batch_size, max_tokens, cache_dir
    )
    return parser.parse()


class _BiocParser:
    def __init__(
        self,
        archives: Iterable[int],
        n_files: int,
        ann_type: str,
        batch_size: int,
        max_tokens: int,
        cache_dir: str | None,
    ):
        self.cache_dir = cache_dir
        self.archives, self.iterfiles = self._get_archives(archives, n_files)
        self.ann_type = ann_type
        self.batch_size = batch_size
        self.max_tokens = max_tokens

        model_id = "allenai/specter"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.mask_token = self.tokenizer.special_tokens_map["mask_token"]
        self.model = AutoModel.from_pretrained(model_id).eval()

        self._ann_ids: list[int] = []
        self._ann_symbols: list[str] = []

    def _get_archives(
        self, file_numbers: Iterable[int], n_files: int
    ) -> tuple[list[str], Callable[[tarfile.TarFile], Sequence[str]]]:
        """Find the local path to the BioC archives."""
        downloader = BiocDownloader(cache_dir=self.cache_dir)
        downloader.file_numbers = file_numbers
        archives = downloader.download()

        def iterfiles(tar: tarfile.TarFile) -> Sequence[str]:
            if n_files < 0:
                return tar.getnames()

            return tar.getnames()[:n_files]

        return (archives, iterfiles)

    def _read_pubmed_data(self):
        # TODO: Assumes genes even if another annotation type is requested.
        pubmed_downloader = PubmedDownloader()
        pubmed_downloader.files = ["gene2pubmed.gz"]
        return pd.read_table(
            pubmed_downloader.download()[0],
            header=0,
            names=["TaxID", "GeneID", "PMID"],
            usecols=["PMID", "GeneID"],
            memory_map=True,
        ).sort_values("PMID")

    def _attach_pubmed_genes(
        self, examples: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        def get_genes(pmid):
            table = self._pubmed_edges
            start = np.searchsorted(table["PMID"], pmid)
            end = start
            while (end < table.shape[0]) and (table["PMID"].iloc[end] == pmid):
                end += 1

            return [table["GeneID"].iloc[idx] for idx in range(start, end)]

        def sorted_in(arr, v) -> bool:
            idx = np.searchsorted(arr, v)
            return idx < len(arr) and arr[idx] == v

        examples["gene2pubmed"] = [
            get_genes(pmid) for pmid in examples["pmid"]
        ]

        # Remove gene IDs not seen in BioC files. Rare but they exist.
        examples["gene2pubmed"] = [
            [g for g in pub_genes if sorted_in(self._ann_ids, g)]
            for pub_genes in examples["gene2pubmed"]
        ]

        return examples

    def parse(self):
        print(f"Reading files from {len(self.archives)} archives.")
        pub_data = list(self._grab_publication_data())
        dataset = datasets.Dataset.from_list(pub_data).map(
            self._tokenize_and_embed,
            batched=True,
            batch_size=self.batch_size,
            with_rank=True,
            remove_columns="abstract",
            num_proc=torch.cuda.device_count(),
            desc="Embed Abstracts",
        )

        features = dataset.features.copy()
        features["embedding"].length = len(dataset["embedding"][0])

        # If first batch of pubmed genes is all empty (unlikely with a large
        # batch), datasets will detect a null feature type instead of using
        # int. Explicitly pass it the correct type to prevent this.
        features["gene2pubmed"] = features["gene2pubtator"]
        self._pubmed_edges = self._read_pubmed_data()
        dataset = dataset.map(
            self._attach_pubmed_genes,
            batched=True,
            features=features,
            desc="Attach pubmed genes",
        )

        def deduplicate(anns: list[str]) -> list[str]:
            values, counts = np.unique_counts(anns)
            duplicated = values[counts > 1]
            counts = counts[counts > 1]

            for i in range(len(anns)):
                if anns[i] in duplicated:
                    loc = np.isin(duplicated, anns[i])
                    anns[i] += f"{counts[loc]}"
                    counts[loc] -= 1

            return anns

        self._ann_symbols = deduplicate(self._ann_symbols)

        features["gene2pubtator"] = datasets.Sequence(
            datasets.ClassLabel(names=self._ann_symbols)
        )
        features["gene2pubmed"] = features["gene2pubtator"]

        def repack_labels(
            examples: dict[str, list[Any]],
        ) -> dict[str, list[Any]]:
            examples["gene2pubtator"] = [
                np.searchsorted(self._ann_ids, ann)
                for ann in (batch for batch in examples["gene2pubtator"])
            ]
            examples["gene2pubmed"] = [
                np.searchsorted(self._ann_ids, ann)
                for ann in (batch for batch in examples["gene2pubmed"])
            ]
            return examples

        return dataset.map(
            repack_labels,
            batched=True,
            features=features,
            desc="Repack label IDs",
        )

    def _grab_publication_data(self) -> dict[str, Any]:
        for i, archive in enumerate(self.archives):
            with tarfile.open(archive, "r") as tar:
                desc = os.path.basename(archive)
                total = len(self.iterfiles(tar))
                with tqdm(total=total, desc=desc, position=i) as pbar:
                    for file in self.iterfiles(tar):
                        try:
                            with tar.extractfile(file) as fd:  # type: ignore[union-attr]
                                for pub_data in self._parse_file(fd):
                                    yield pub_data
                        except ET.ParseError as e:
                            print(f"Failed to parse {file}:\n  {e.msg}")

                        pbar.update()

    def _parse_file(self, fd) -> Iterable[dict[str, Any]]:
        tree = ET.parse(fd)
        for doc in tree.iterfind("document"):
            try:
                pmid, abstract, year, annotations = self._parse_doc(doc)

                for ann_id, symbol in annotations:
                    idx = np.searchsorted(self._ann_ids, ann_id)
                    if (
                        idx >= len(self._ann_ids)
                        or ann_id != self._ann_ids[idx]
                    ):
                        self._ann_ids.insert(idx, ann_id)
                        self._ann_symbols.insert(idx, symbol or "")
                    elif symbol and (
                        (not self._ann_symbols[idx])
                        or (len(symbol) < len(self._ann_symbols[idx]))
                    ):
                        # Prefer smaller symbols
                        self._ann_symbols[idx] = symbol

                yield {
                    "pmid": pmid,
                    "abstract": abstract,
                    "year": year,
                    "gene2pubtator": list(
                        {ann_id for ann_id, _ in annotations}
                    ),
                }
            except ValueError:
                continue

    def _parse_doc(
        self, doc: ET.Element
    ) -> tuple[int, str, int | None, list[tuple[int, str | None]]]:
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

        abstracts_elements = self._get_abstracts(doc)
        stripped_abstract = "".join(
            self._mask_abstract(abstract) for abstract in abstracts_elements
        )

        annotations = [
            (ann_id, ann_symbol)
            for ann_id, ann_symbol in sum(
                (
                    self._collect_annotations(abstract)
                    for abstract in abstracts_elements
                ),
                [],
            )
            if (ann_id is not None)
        ]

        return (int(pmid), stripped_abstract, year, annotations)

    def _get_abstracts(self, doc: ET.Element) -> list[ET.Element]:
        return [
            passage
            for passage in doc.iterfind("passage")
            if any(
                (
                    (
                        infon.attrib["key"] == "type"
                        and infon.text == "abstract"
                    )
                    for infon in passage.iterfind("infon")
                )
            )
        ]

    def _is_type(self, annotation: ET.Element, ann_type: str) -> bool:
        return any(
            (
                (infon.attrib["key"] == "type") and (infon.text == ann_type)
                for infon in annotation.iterfind("infon")
            )
        )

    def _get_identifier(
        self,
        annotation: ET.Element,
    ) -> tuple[int | None, str | None]:
        for infon in annotation.iterfind("infon"):
            if infon.attrib["key"] == "identifier":
                symbol = annotation.find("text")
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

                return (ann_id, symbol.text if symbol is not None else None)

        return (None, None)

    def _mask_abstract(self, passage: ET.Element) -> str:
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
            if not self._is_type(ann, self.ann_type):
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
                letter if mask[i] > 0 else self.mask_token
                for i, letter in enumerate(text)
                if mask[i]
            )
        )

    def _collect_annotations(
        self, passage: ET.Element
    ) -> list[tuple[int | None, str | None]]:
        return [
            self._get_identifier(ann)
            for ann in passage.iterfind("annotation")
            if self._is_type(ann, self.ann_type)
        ]

    def _tokenize_and_embed(
        self, examples: dict[str, Any], rank: int
    ) -> dict[str, Any]:
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        self.model.to(device)
        inputs = self.tokenizer(
            examples["abstract"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        ).to(device)

        with torch.no_grad():
            return {
                "embedding": self.model(**inputs).last_hidden_state[:, 0, :]
            }
