"""UI components for the webapp."""

__all__ = ["home", "pmid_search_page", "results", "search_pmid"]

import re
from datetime import datetime

from fastapi import Request
from fastapi.templating import Jinja2Templates
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter

from webapp import config as cfg
from webapp import query
from webapp.genes import Gene, top_predictions

templates = Jinja2Templates(directory="webapp/templates")


def _top_preds(predictions: list[float], genes: Gene) -> dict[str, list[str]]:
    return top_predictions(
        predictions, genes, k=cfg.min_genes, p=cfg.gene_thresh
    )


def _format_abstract(abstract: str) -> str:
    return re.sub(
        r" ([A-Z /]{4,}:)",
        r'<p/><p class="abstract">\1',
        '<p class="abstract">' + abstract + "</p>",
    )


def home(request: Request, min_year: int, n_publications: int):
    year_range = {"min_year": min_year, "max_year": datetime.today().year}
    n_publications = round(n_publications / 1e5)

    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={
            "year_range": year_range,
            "n_publications": f"{n_publications / 10}",
        },
    )


def pmid_search_page(request: Request, min_year: int, n_publications: int):
    year_range = {"min_year": min_year, "max_year": datetime.today().year}
    n_publications = round(n_publications / 1e5)

    return templates.TemplateResponse(
        request=request,
        name="search_pmid.html",
        context={
            "year_range": year_range,
            "n_publications": f"{n_publications / 10}",
        },
    )


def _extract_points(points, scores=None):
    if not scores:
        scores = [None] * len(points)

    return [
        {
            k: (
                point.id
                if k == "pmid"
                else (
                    score or point.score
                    if k == "similarity"
                    else (
                        len(point.payload[k])
                        if k == "reference"
                        else point.payload[k]
                    )
                )
            )
            for k in (
                "abstract",
                "title",
                "year",
                "pmid",
                "similarity",
                "reference",
                "pubtator3_genes",
            )
        }
        for point, score in zip(points, scores)
        if point.payload
    ]


async def results(
    request: Request,
    client: AsyncQdrantClient,
    session_id: str,
    year: tuple[int, int],
    behavioral: bool,
    molecular: bool,
    page: int,
    genes: Gene,
    collection_name: str,
):
    point = await client.retrieve(
        cfg.tmp_collection_name,
        [session_id],
        with_payload=True,
        with_vectors=True,
    )

    if (len(point) == 0) or (not point[0].payload):
        return f"Server error: session {session_id} not cached."

    prediction = point[0].vector
    title = point[0].payload["title"]
    abstract = point[0].payload["abstract"]

    points = await query.search_with_abstract(
        client,
        prediction,
        title,
        abstract,
        year,
        behavioral,
        molecular,
        page,
        collection_name,
    )

    n_points = await client.count(
        collection_name,
        count_filter=Filter(
            must=query.query_filters(year, behavioral, molecular),
        ),
        exact=False,
    )
    last_page = (page * cfg.results_per_page) >= n_points.count

    top_genes = _top_preds(prediction, genes)

    results = _extract_points(points)
    for i, pt in enumerate(points):
        results[i]["genes"] = _top_preds(pt.vector, genes)
        results[i]["abstract"] = _format_abstract(results[i]["abstract"])

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Publications similar to:",
            "parent": {
                "title": title,
                "abstract": _format_abstract(abstract),
                "genes": top_genes,
            },
            "results": results,
            "year_range": {"min_year": year[0], "max_year": year[1]},
            "page": page,
            "last_page": last_page,
            "session_id": session_id,
        },
    )


async def search_pmid(
    request: Request,
    client: AsyncQdrantClient,
    positive: list[int],
    negative: list[int],
    year: tuple[int, int],
    behavioral: bool,
    molecular: bool,
    page: int,
    genes: Gene,
    collection_name: str,
):
    if not positive:
        return "Must provide at least one positive example."

    positive_points = await client.retrieve(
        collection_name, ids=positive, with_payload=True, with_vectors=True
    )

    negative_points = []
    if negative:
        negative_points = await client.retrieve(
            collection_name,
            ids=negative,
            with_payload=True,
            with_vectors=False,
        )

    found = {int(point.id) for point in (positive_points + negative_points)}
    missing = {*positive, *negative} - found

    if missing:
        return f"Publications with PMID: {missing} not in database"

    if (len(positive_points) == 0) or (positive_points[0].payload is None):
        return f"No publication with PMID: {positive[0]} in database."

    main_point = positive_points[0]
    top_genes = _top_preds(main_point.vector, genes)

    query_filter = query.query_filters(year, behavioral, molecular)
    points = await client.recommend(
        collection_name,
        positive,
        negative,
        with_payload=True,
        with_vectors=True,
        query_filter=Filter(must=query_filter),
        limit=cfg.results_per_page,
        offset=(page - 1) * cfg.results_per_page,
    )

    n_points = await client.count(
        collection_name,
        count_filter=Filter(must=query_filter),
        exact=False,
    )
    last_page = (page * cfg.results_per_page) >= n_points.count

    results = _extract_points(points)
    for i, pt in enumerate(points):
        results[i]["genes"] = _top_preds(pt.vector, genes)
        results[i]["abstract"] = _format_abstract(results[i]["abstract"])

    parent = {
        "title": main_point.payload["title"],
        "abstract": _format_abstract(main_point.payload["abstract"]),
        "genes": top_genes,
    }

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Publications similar to:",
            "parent": parent,
            "results": results,
            "year_range": {"min_year": year[0], "max_year": year[1]},
            "page": page,
            "last_page": last_page,
        },
    )


async def analyze_references(
    request: Request,
    client: AsyncQdrantClient,
    pmid: int,
    behavioral: bool,
    molecular: bool,
    genes: Gene,
    collection_name: str,
):
    results = await query.analyze_references(
        client, pmid, behavioral, molecular, collection_name
    )

    parent = {
        "title": results["parent"].payload["title"],
        "abstract": _format_abstract(results["parent"].payload["abstract"]),
        "genes": _top_preds(results["parent"].vector, genes),
    }

    if len(results["references"]) == 0:
        return f"No references for {pmid} found in database."

    references = _extract_points(results["references"], results["scores"])
    references = sorted(
        references, key=lambda x: x["similarity"], reverse=True
    )

    for i, pt in enumerate(results["references"]):
        references[i]["genes"] = _top_preds(pt.vector, genes)
        references[i]["abstract"] = _format_abstract(references[i]["abstract"])

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Reference similarity for:",
            "parent": parent,
            "results": references,
            "year_range": {"min_year": 0, "max_year": 2025},
            "page": 1,
            "last_page": True,
        },
    )
