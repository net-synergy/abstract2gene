"""Methods for querying and searching the database."""

__all__ = ["search_with_abstract", "get_min_year", "query_filters"]

from typing import Sequence

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Range,
    ScoredPoint,
    ValuesCount,
)

from webapp import config as cfg


async def get_min_year(client: AsyncQdrantClient, collection_name: str) -> int:
    """Find the earliest publication year in the database."""
    scroll = await client.scroll(
        collection_name=collection_name, with_payload=True, with_vectors=False
    )

    return min((point.payload["year"] for point in scroll[0] if point.payload))


def query_filters(
    year: tuple[int, int] | None,
    behavioral: bool = True,
    molecular: bool = True,
) -> Sequence[FieldCondition]:
    query_filter = []

    if year is not None:
        query_filter.append(
            FieldCondition(key="year", range=Range(gte=year[0], lte=year[1]))
        )

    if molecular and (not behavioral):
        query_filter.append(
            FieldCondition(
                key="pubtator3_genes", values_count=ValuesCount(gt=0)
            )
        )
    elif behavioral and (not molecular):
        query_filter.append(
            FieldCondition(
                key="pubtator3_genes", values_count=ValuesCount(lte=0)
            )
        )
    elif not (behavioral or molecular):
        # A count can never be less than 0 so this should return no results.
        query_filter = [
            FieldCondition(
                key="pubtator3_genes", values_count=ValuesCount(lt=0)
            )
        ]

    return query_filter


async def search_with_abstract(
    client: AsyncQdrantClient,
    prediction: list[float],
    title: str,
    abstract: str,
    year: tuple[int, int],
    behavioral: bool,
    molecular: bool,
    page: int,
    collection_name: str,
) -> list[ScoredPoint]:
    """Find publications with similar genetic components to an abstract."""
    query_filter = query_filters(year, behavioral, molecular)
    return await client.search(
        collection_name=collection_name,
        query_vector=prediction,
        query_filter=Filter(must=query_filter),
        with_payload=True,
        with_vectors=True,
        limit=cfg.results_per_page,
        offset=(page - 1) * cfg.results_per_page,
    )


async def analyze_references(
    client: AsyncQdrantClient,
    pmid: int,
    behavioral: bool,
    molecular: bool,
    collection_name: str,
):
    """Analyze genetic similarity between a publication and its references."""
    parent_records = await client.retrieve(
        collection_name=collection_name,
        ids=[pmid],
        with_payload=True,
        with_vectors=True,
    )
    parent = parent_records[0]

    references = await client.retrieve(
        collection_name=collection_name,
        ids=parent.payload["reference"],
        with_payload=True,
        with_vectors=True,
    )

    if len(references) == 0:
        return {"parent": parent, "references": [], "scores": []}

    if not (behavioral and molecular):
        if behavioral:
            references = [
                ref
                for ref in references
                if len(ref.payload["pubtator3_genes"]) == 0
            ]
        else:
            references = [
                ref
                for ref in references
                if len(ref.payload["pubtator3_genes"]) > 0
            ]

    parent_vec = np.asarray(parent.vector)
    parent_vec = parent_vec / np.linalg.norm(parent_vec)
    ref_vecs = np.vstack([np.asarray(ref.vector) for ref in references])
    ref_vecs = ref_vecs / np.linalg.norm(ref_vecs, axis=1, keepdims=True)

    scores = (ref_vecs @ parent_vec).tolist()

    return {"parent": parent, "references": references, "scores": scores}
