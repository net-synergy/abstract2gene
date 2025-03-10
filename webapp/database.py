"""Connect to and build the qdrant database."""

from typing import Any

import datasets
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import abstract2gene as a2g


def connect() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def _generate_points(
    examples: dict[str, list[Any]], model: a2g.model.Model
) -> dict[str, Any]:
    abstracts = [
        title + "[SEP]" + abstract
        for title, abstract in zip(examples["title"], examples["abstract"])
    ]
    return {"prediction": list(model.predict(abstracts))}


def init_db(
    client: QdrantClient, model: a2g.model.Model, collection_name: str
):
    """Set up the gene vector collection."""
    if model.templates is None:
        raise RuntimeError("Templates not attached to this model.")

    n_genes = model.templates.indices.shape[0]
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=n_genes, distance=Distance.COSINE),
    )


def store_publications(
    client: QdrantClient,
    dataset: datasets,
    model: a2g.model.Model,
    collection_name: str,
):
    """Add publications from a dataset to the vector collection."""
    dataset = dataset.map(
        _generate_points,
        fn_kwargs={"model": model},
        batched=True,
        batch_size=1000,
    )

    points = [
        PointStruct(
            id=example["pmid"],
            vector=list(example["prediction"]),
            payload={
                "year": example["year"],
                "title": example["title"],
                "abstract": example["abstract"],
                "pubtator3_genes": example["gene"],
                "reference": example["reference"],
            },
        )
        for example in dataset
    ]

    client.upsert(collection_name=collection_name, wait=True, points=points)
