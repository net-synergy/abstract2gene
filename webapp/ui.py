"""UI components for the webapp."""

__all__ = ["home", "pmid_search_page", "results", "search_pmid"]

from datetime import datetime

from fastapi import Request
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Range

import abstract2gene as a2g
from webapp import query

templates = Jinja2Templates(directory="webapp/templates")


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
            )
        }
        for point in points
        if point.payload
    ]


def results(
    request: Request,
    client: QdrantClient,
    model: a2g.model.Model,
    title: str,
    abstract: str,
    year: tuple[int, int],
    collection_name: str,
):
    points = query.search_with_abstract(
        client, model, title, abstract, year, collection_name
    )

    if points is None or len(points) == 0:
        return "No results found."

    results = _extract_points(points)
    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Publications similar to: ",
            "parent": {"title": title, "abstract": abstract},
            "results": results,
            "year_range": {"min_year": year[0], "max_year": year[1]},
        },
    )


def search_pmid(
    request: Request,
    client: QdrantClient,
    positive: list[int],
    negative: list[int],
    year: tuple[int, int],
    collection_name: str,
):
    if not positive:
        return "Must provide at least one positive example."

    positive_points = client.retrieve(
        collection_name, ids=positive, with_payload=True, with_vectors=False
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

    points = client.recommend(
        collection_name,
        positive,
        negative,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="year", range=Range(gte=year[0], lte=year[1])
                )
            ]
        ),
        ## TODO: Come up with p-value based method for determining which
        ## references to return instead of static limit.
        limit=10,
    )

    if points is None or len(points) == 0:
        return "No results found."

    results = _extract_points(points)
    parent = {
        "title": main_point.payload["title"],
        "abstract": main_point.payload["abstract"],
    }

    return templates.TemplateResponse(
        request,
        name="a2g_results.html",
        context={
            "action_title": "Publications similar to: ",
            "parent": parent,
            "results": results,
            "year_range": {"min_year": year[0], "max_year": year[1]},
        },
    )


def analyze_references(
    request: Request, client: QdrantClient, pmid: int, collection_name: str
):
    results = query.analyze_references(client, pmid, collection_name)
    parent = {
        "title": results["parent"].payload["title"],
        "abstract": results["parent"].payload["abstract"],
    }

    if len(results["references"]) == 0:
        return f"No references for {pmid} found in database."

    references = _extract_points(results["references"])
    references = sorted(
        references, key=lambda x: x["similarity"], reverse=True
    )

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
