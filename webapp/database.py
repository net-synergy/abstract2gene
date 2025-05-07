"""Connect to and build the qdrant database."""

import os
from typing import Any

import datasets
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

import abstract2gene as a2g


def connect() -> AsyncQdrantClient:
    qdrant_domain = os.getenv("A2G_QDRANT_URL") or "localhost"
    return AsyncQdrantClient(url=f"http://{qdrant_domain}:6333")


def _generate_points(
    examples: dict[str, list[Any]], model: a2g.model.Model
) -> dict[str, Any]:
    abstracts = [
        title + "[SEP]" + abstract
        for title, abstract in zip(examples["title"], examples["abstract"])
    ]
    return {"prediction": list(model.predict(abstracts))}


async def init_db(
    client: AsyncQdrantClient, model: a2g.model.Model, collection_name: str
):
    """Set up the gene vector collection."""
    if model.templates is None:
        raise RuntimeError("Templates not attached to this model.")

    n_genes = model.templates.names.shape[0]
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=n_genes,
            distance=Distance.COSINE,
            on_disk=True,
            datatype="float16",
        ),
        on_disk_payload=True,
    )


async def store_publications(
    client: AsyncQdrantClient,
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
        desc="Predicting genes",
    )

    batch_size = 500
    for i in tqdm(range(0, len(dataset), batch_size)):
        fin = min(i + batch_size, len(dataset))
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
            for example in dataset.select(range(i, fin))
        ]

        await client.upsert(
            collection_name=collection_name,
            wait=False,
            points=points,
        )


async def store_user_abstracts(
    client: AsyncQdrantClient,
    model: a2g.model.Model,
    title: str,
    abstract: str,
    session_id: str,
    collection_name: str,
):
    prediction = model.predict(title + "[SEP]" + abstract).tolist()[0]

    point = [
        PointStruct(
            id=session_id,
            vector=prediction,
            payload={
                "title": title,
                "abstract": abstract,
            },
        )
    ]

    await client.upsert(
        collection_name=collection_name,
        wait=True,
        points=point,
    )
