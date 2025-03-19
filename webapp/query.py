"""Methods for querying and searching the database."""

__all__ = ["search_with_abstract", "get_min_year"]

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Range, ScoredPoint

import abstract2gene as a2g
from webapp import config as cfg


def get_min_year(client: QdrantClient, collection_name: str) -> int:
    """Find the earliest publication year in the database."""
    scroll = client.scroll(
        collection_name=collection_name, with_payload=True, with_vectors=False
    )[0]

    return min((point.payload["year"] for point in scroll if point.payload))


def search_with_abstract(
    client: QdrantClient,
    prediction: list[float],
    title: str,
    abstract: str,
    year: tuple[int, int],
    page: int,
    collection_name: str,
) -> tuple[list[float], list[ScoredPoint]]:
    """Find publications with similar genetic components to an abstract."""
    return (
        prediction,
        client.search(
            collection_name=collection_name,
            query_vector=prediction,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="year", range=Range(gte=year[0], lte=year[1])
                    )
                ]
            ),
            with_payload=True,
            with_vectors=True,
            limit=cfg.results_per_page,
            offset=(page - 1) * cfg.results_per_page,
        ),
    )


def _references_in_db(
    client: QdrantClient, collection_name: str, ref_list: list[int]
) -> list[int]:
    """Filter reference_list to PMIDs in the database."""
    n_points = client.get_collection(collection_name).points_count
    if not n_points:
        raise RuntimeError("Qdrant database is empty")

    all_refs = [
        ref.id
        for ref in client.scroll(
            collection_name,
            limit=n_points,
            with_payload=False,
            with_vectors=False,
        )[0]
    ]
    all_refs.sort()

    return [
        ref
        for ref in ref_list
        if all_refs[np.searchsorted(all_refs, ref)] == ref
    ]


def analyze_references(
    client: QdrantClient,
    pmid: int,
    collection_name: str,
):
    """Analyze genetic similarity between a publication and its references."""
    parent = client.retrieve(
        collection_name=collection_name,
        ids=[pmid],
        with_payload=True,
        with_vectors=True,
    )[0]

    ref_ids = _references_in_db(
        client, collection_name, parent.payload["reference"]
    )

    if len(ref_ids) == 0:
        return {"parent": parent, "references": [], "scores": []}

    references = client.retrieve(
        collection_name=collection_name,
        ids=ref_ids,
        with_payload=True,
        with_vectors=True,
    )

    parent_vec = np.asarray(parent.vector)[None, :]
    parent_vec = parent_vec / np.linalg.norm(parent_vec, axis=1, keepdims=True)
    ref_vecs = np.vstack([np.asarray(ref.vector) for ref in references])
    parent_vec = ref_vecs / np.linalg.norm(ref_vecs, axis=1, keepdims=True)

    scores = (parent_vec @ ref_vecs.T).tolist()

    return {"parent": parent, "references": references, "scores": scores}
