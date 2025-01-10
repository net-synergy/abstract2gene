"""Collect files from BioCXML as DataSets and Templates."""

__all__ = ["bioc2dataset"]

import contextlib
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModel

from abstract2gene.data import BiocDownloader

from ._dataset import DataSet


def bioc2dataset(
    archives: Iterable[int],
    n_files: int = -1,
    ann_type="Gene",
    embed_bs: int = 50,
    max_tokens: int = 512,
    **kwds,
) -> DataSet:
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
    embed_bs : int, default 50
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
    **kwds : dict[str, Any]
        Keyword arguments to be passed to the DataSet constructor.

    Returns
    -------
    dataset : a2g.dataset.DataSet
        A dataset containing the abstract embeddings for each publication and a
        list of labels.

    """
    parser = _BiocParser(archives, n_files, ann_type, embed_bs, max_tokens)
    parser.parse()
    features, labels = parser.to_arrays()
    return DataSet(
        features,
        labels,
        parser.sample_names(),
        parser.annotation_names(),
        **kwds,
    )


class _BiocParser:
    def __init__(
        self,
        archives: Iterable[int],
        n_files: int,
        ann_type: str,
        batch_size: int,
        max_tokens: int,
    ):
        self.archives, self.iterfiles = self._get_archives(archives, n_files)
        self.ann_type = ann_type
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.model = jax.jit(FlaxAutoModel.from_pretrained("allenai/specter"))

        self._features: list[jnp.ndarray] = []
        self._pmid_names: list[str] = []
        self._edge_list: list[list[int]] = []
        self._ann_ids: list[int] = []
        self._ann_symbols: list[str] = []

    def _get_archives(
        self, file_numbers: Iterable[int], n_files: int
    ) -> tuple[list[str], Callable[[tarfile.TarFile], Iterable[str]]]:
        """Find the local path to the BioC archives."""
        downloader = BiocDownloader(check_remote=False)
        downloader.file_numbers = file_numbers
        archives = downloader.download()

        def iterfiles(tar: tarfile.TarFile) -> Iterable[str]:
            # TMP: Drop files with 6 digits in the name since these (seem to)
            # have 1000 docs per file instead of 100. It's throwing off tqdm's
            # time predictions.
            names = [
                n for n in tar.getnames() if len(re.findall(r"\d{6}", n)) == 0
            ]
            if n_files < 0:
                return names

            return names[:n_files]

        return (archives, iterfiles)

    def parse(self):
        print(f"Starting embedding for {len(self.archives)} archives.")
        for archive in self.archives:
            with tarfile.open(archive, "r") as tar:
                desc = os.path.basename(archive)
                total = len(self.iterfiles(tar))
                with tqdm(total=total, desc=desc) as pbar:
                    for file in self.iterfiles(tar):
                        try:
                            with tar.extractfile(file) as fd:
                                abstracts = self._parse_file(fd)
                            self._features.extend(self._embed(abstracts))
                        except ET.ParseError as e:
                            print(f"Failed to parse {file}:\n  {e.msg}")
                        pbar.update()

    def _parse_file(self, fd) -> list[str]:
        tree = ET.parse(fd)
        abstracts: list[str] = []
        for doc in tree.findall("document"):
            try:
                pmid, abstract, annotations = self._parse_doc(doc)
            except ValueError:
                continue

            self._edge_list.append([ann_id for ann_id, _ in annotations])
            self._pmid_names.append(pmid)
            abstracts.append(abstract)

            for ann_id, symbol in annotations:
                idx = np.searchsorted(self._ann_ids, ann_id)
                if idx >= len(self._ann_ids) or ann_id != self._ann_ids[idx]:
                    self._ann_ids.insert(idx, ann_id)
                    self._ann_symbols.insert(idx, symbol)
                elif len(symbol) < len(self._ann_symbols[idx]):
                    # Prefer smaller symbols
                    self._ann_symbols[idx] = symbol

        return abstracts

    def _parse_doc(
        self, doc: ET.Element
    ) -> tuple[str, str, list[tuple[int, str]]]:
        id_field = doc.find("id")

        # I don't think this should ever happen so raising an error to make it
        # obvious if this scenario occurs.
        if id_field is None:
            raise ValueError("Document missing ID field.")

        pmid = id_field.text
        if pmid is None:
            raise ValueError("Document missing ID field.")

        abstracts_elements = self._get_abstracts(doc)
        stripped_abstract = "".join(
            self._strip_abstract(abstract) for abstract in abstracts_elements
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
            if (ann_id is not None) and (ann_symbol is not None)
        ]

        return (pmid, stripped_abstract, annotations)

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

                return (
                    ann_id,
                    symbol.text if symbol is not None else None,
                )

        return (None, None)

    def _strip_abstract(self, passage: ET.Element) -> str:
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
            mask[pos_start:pos_end] = 0

        return "".join((letter for i, letter in enumerate(text) if mask[i]))

    def _collect_annotations(
        self, passage: ET.Element
    ) -> list[tuple[int | None, str | None]]:
        return [
            self._get_identifier(ann)
            for ann in passage.iterfind("annotation")
            if self._is_type(ann, self.ann_type)
        ]

    def _embed(self, abstracts: list[str]) -> list[jnp.ndarray]:
        start_idx = 0
        end_idx = min(self.batch_size, len(abstracts))

        features: list[jnp.ndarray] = []
        while start_idx < end_idx:
            inputs = self.tokenizer(
                abstracts[start_idx:end_idx],
                return_tensors="jax",
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
            )
            outputs = self.model(**inputs)
            features.append(outputs.last_hidden_state[:, 0, :])

            start_idx = end_idx
            end_idx = min(end_idx + self.batch_size, len(abstracts))

        return features

    def to_arrays(
        self,
    ) -> tuple[jax.Array, np.ndarray[Any, np.dtype[np.bool]]]:
        if len(self._features) == 0:
            raise ValueError(
                """No features were successfully created.

                This was likely caused by all provided bioc files having parse
                errors.
                """
            )
        n_edges = sum((len(anns) for anns in self._edge_list))
        data = np.ones((n_edges,), dtype=np.bool)
        rows = np.zeros((n_edges,))
        cols = np.zeros((n_edges,))
        shape = (len(self._pmid_names), len(self._ann_ids))
        count = 0
        for i, anns in enumerate(self._edge_list):
            for ann in anns:
                rows[count] = i
                # Convert original IDs to position. This drops information but
                # original IDs may have large gaps between them making the
                # number of columns huge. This doesn't directly matter for
                # sparse arrays but it makes masks significantly larger than
                # needed.
                cols[count] = np.searchsorted(self._ann_ids, ann)
                count += 1

        return (
            jnp.concat(self._features),
            sp.sparse.coo_array((data, (rows, cols)), shape).tocsc(),
        )

    def sample_names(self) -> np.ndarray[Any, np.dtype[np.str_]]:
        return np.asarray(self._pmid_names)

    def annotation_names(self) -> np.ndarray[Any, np.dtype[np.str_]]:
        return np.asarray(self._ann_symbols)
