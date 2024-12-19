"""Collect files from BioCXML as DataSets and Templates."""

__all__ = ["bioc2dataset"]

import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModel

from abstract2gene.data import BiocDownloader

from ._dataset import DataSet


def bioc2dataset(
    files: Iterable[int],
    ann_type="Gene",
    embed_bs: int = 50,
    max_tokens: int = 512,
    min_occurances: int = 50,
    **kwds,
) -> DataSet:
    parser = _BiocParser(files, ann_type, embed_bs, max_tokens, min_occurances)
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
        files: Iterable[int],
        ann_type: str,
        batch_size: int,
        max_tokens: int,
        min_occurances: int,
    ):
        self.archives = self._tar2files(files)
        self.ann_type = ann_type
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.min_occurances = min_occurances
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.model = jax.jit(FlaxAutoModel.from_pretrained("allenai/specter"))
        self._pmid2features: dict[str, jnp.ndarray] = {}
        self._pmid2ann: dict[str, list[str]] = {}
        self._pub2idx: dict[str, int] = {}
        self._ann2idx: dict[str, int] = {}
        self._ann_occurances: dict[str, int] = defaultdict(int)
        self._ann_count = 0
        self._pub_count = 0
        self._idx2symbol: dict[int, str] = {}

    def _tar2files(self, files: Iterable[int]) -> dict[str, list[int]]:
        """Generate dictionary mapping tar files to list of files contained."""
        file_map = defaultdict(list)
        for f in files:
            file_map[int(str(f)[-1])].append(f)
        downloader = BiocDownloader()
        downloader.file_numbers = set(file_map.keys())
        archives = downloader.download()

        out = {}
        for i in file_map:
            archive = [
                archive
                for archive in archives
                if archive.endswith(f"{i}.tar.gz")
            ][0]
            out[archive] = file_map[i]

        return out

    def parse(self):
        file_template = "output/BioCXML/{}.BioC.XML"
        total_files = sum((len(fs) for fs in self.archives.values()))

        with tqdm(total=total_files, desc="Specter") as pbar:
            for archive, files in self.archives.items():
                with tarfile.open(archive, "r") as tar:
                    for file in files:
                        fd = tar.extractfile(file_template.format(file))
                        try:
                            pmids, abstracts = self._parse_file(fd)
                            self._embed(pmids, abstracts, pbar)
                        except ValueError:
                            pass

                        fd.close()
                        pbar.update()

    def _parse_file(self, fd) -> tuple[list[str], list[str]]:
        tree = ET.parse(fd)
        pmids: list[str] = []
        abstracts: list[str] = []
        for doc in tree.findall("document"):
            pmid, abstract, annotations = self._parse_doc(doc)
            self._pub2idx[pmid] = self._pub_count
            self._pub_count += 1
            self._pmid2ann[pmid] = [ann_id for ann_id, _ in annotations]
            for ann_id, _ in annotations:
                self._ann_occurances[ann_id] += 1
            pmids.append(pmid)
            abstracts.append(abstract)

            for ann_id, symbol in annotations:
                if (
                    (ann_id is not None)
                    and (symbol is not None)
                    and (ann_id not in self._ann2idx)
                ):
                    self._ann2idx[ann_id] = self._ann_count
                    self._idx2symbol[self._ann_count] = symbol
                    self._ann_count += 1
                elif (
                    (ann_id is not None)
                    and (symbol is not None)
                    and (
                        len(symbol)
                        < len(self._idx2symbol[self._ann2idx[ann_id]])
                    )
                ):
                    # Prefer abbreviations to full name.
                    self._idx2symbol[self._ann2idx[ann_id]] = symbol

        return (pmids, abstracts)

    def _parse_doc(
        self, doc: ET.Element
    ) -> tuple[str, str, list[tuple[str, str]]]:
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
    ) -> tuple[str | None, str | None]:
        for infon in annotation.iterfind("infon"):
            if infon.attrib["key"] == "identifier":
                symbol = annotation.find("text")
                return (
                    infon.text,
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
    ) -> list[tuple[str | None, str | None]]:
        return [
            self._get_identifier(ann)
            for ann in passage.iterfind("annotation")
            if self._is_type(ann, self.ann_type)
        ]

    def _embed(
        self, pmids: list[str], abstracts: list[str], pbar: tqdm
    ) -> None:
        start_idx = 0
        end_idx = min(self.batch_size, len(abstracts))

        while start_idx < end_idx:
            inputs = self.tokenizer(
                abstracts[start_idx:end_idx],
                return_tensors="jax",
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
            )
            outputs = self.model(**inputs)
            for pmid, embedding in zip(
                pmids[start_idx:end_idx], outputs.last_hidden_state[:, 0, :]
            ):
                self._pmid2features[pmid] = embedding.reshape((1, -1))

            start_idx = end_idx
            end_idx = min(end_idx + self.batch_size, len(abstracts))

    def to_arrays(
        self,
    ) -> tuple[jax.Array, np.ndarray[Any, np.dtype[np.bool]]]:
        ann_mask = (
            np.asarray(list(self._ann2idx.values())) >= self.min_occurances
        )
        indices = np.cumsum(ann_mask)
        labels = np.zeros(
            (len(self._pub2idx), ann_mask.sum()), np.dtype(jnp.bool)
        )
        for pmid, anns in self._pmid2ann.items():
            for ann in anns:
                if self._ann_occurances[ann] < self.min_occurances:
                    continue

                labels[self._pub2idx[pmid], indices[self._ann2idx[ann]]] = True

        return (
            jnp.concat(tuple(self._pmid2features.values()), axis=0),
            labels,
        )

    def sample_names(self) -> np.ndarray[Any, np.dtype[np.str_]]:
        return np.asarray(list(self._pub2idx.keys()))

    def annotation_names(self) -> np.ndarray[Any, np.dtype[np.str_]]:
        ann_mask = (
            np.asarray(list(self._ann2idx.values())) >= self.min_occurances
        )
        return np.asarray(list(self._idx2symbol.values()))[ann_mask]
