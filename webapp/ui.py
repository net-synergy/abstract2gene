"""UI components for the webapp."""

__all__ = ["home", "pmid_search_page", "results", "search_pmid"]

from datetime import datetime

from fastapi import Request
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

import abstract2gene as a2g
from webapp import config as cfg
from webapp import query
from webapp.genes import Gene, top_predictions

templates = Jinja2Templates(directory="webapp/templates")


def _top_preds(predictions: list[float], genes: Gene) -> dict[str, list[str]]:
    return top_predictions(
        predictions, genes, k=cfg.min_genes, p=cfg.gene_thresh
    )


def home(request: Request, min_year: int):
    year_range = {"min_year": min_year, "max_year": datetime.today().year}
    return templates.TemplateResponse(
        request=request, name="search.html", context={"year_range": year_range}
    )


def pmid_search_page(request: Request, min_year: int):
    year_range = {"min_year": min_year, "max_year": datetime.today().year}
    return templates.TemplateResponse(
        request=request,
        name="search_pmid.html",
        context={"year_range": year_range},
    )


def _extract_points(points):
    return [
        {
            k: (
                point.id
                if k == "pmid"
                else (
                    point.score
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
        for point in points
        if point.payload
    ]


def results(
    request: Request,
    client: QdrantClient,
    session_id: str,
    year: tuple[int, int],
    behavioral: bool,
    molecular: bool,
    page: int,
    genes: Gene,
    collection_name: str,
):
    point = client.retrieve(
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

    prediction, points = query.search_with_abstract(
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

    last_page = (page * cfg.results_per_page) >= client.count(
        collection_name,
        count_filter=Filter(
            must=query.query_filters(year, behavioral, molecular),
        ),
    ).count

    top_genes = _top_preds(prediction, genes)

    results = _extract_points(points)
    for i, pt in enumerate(points):
        results[i]["genes"] = _top_preds(pt.vector, genes)

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Publications similar to: ",
            "parent": {
                "title": title,
                "abstract": abstract,
                "genes": top_genes,
            },
            "results": results,
            "year_range": {"min_year": year[0], "max_year": year[1]},
            "page": page,
            "last_page": last_page,
            "session_id": session_id,
        },
    )


def search_pmid(
    request: Request,
    client: QdrantClient,
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

    positive_points = client.retrieve(
        collection_name, ids=positive, with_payload=True, with_vectors=True
    )

    negative_points = []
    if negative:
        negative_points = client.retrieve(
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
    points = client.recommend(
        collection_name,
        positive,
        negative,
        with_payload=True,
        with_vectors=True,
        query_filter=Filter(must=query_filter),
        limit=cfg.results_per_page,
        offset=(page - 1) * cfg.results_per_page,
    )
    last_page = (page * cfg.results_per_page) >= client.count(
        collection_name,
        count_filter=Filter(must=query_filter),
    ).count

    results = _extract_points(points)
    for i, pt in enumerate(points):
        results[i]["genes"] = _top_preds(pt.vector, genes)

    parent = {
        "title": main_point.payload["title"],
        "abstract": main_point.payload["abstract"],
        "genes": top_genes,
    }

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Publications similar to: ",
            "parent": parent,
            "results": results,
            "year_range": {"min_year": year[0], "max_year": year[1]},
            "page": page,
            "last_page": last_page,
        },
    )


def analyze_references(
    request: Request,
    client: QdrantClient,
    pmid: int,
    behavioral: bool,
    molecular: bool,
    genes: Gene,
    collection_name: str,
):
    results = query.analyze_references(
        client, pmid, behavioral, molecular, collection_name
    )
    parent = {
        "title": results["parent"].payload["title"],
        "abstract": results["parent"].payload["abstract"],
        "genes": _top_preds(results["parent"].vector, genes),
    }

    if len(results["references"]) == 0:
        return f"No references for {pmid} found in database."

    references = _extract_points(results["references"])
    references = sorted(
        references, key=lambda x: x["similarity"], reverse=True
    )

    for i, pt in enumerate(results["references"]):
        references[i]["genes"] = _top_preds(pt.vector, genes)

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Reference similarity for",
            "parent": parent,
            "results": references,
            "year_range": {"min_year": 0, "max_year": 2025},
        },
    )
