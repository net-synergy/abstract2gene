"""Entry point for the search engine's web UI."""

import json
import os
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import AsyncQdrantClient

import abstract2gene as a2g
import webapp.config as cfg
from abstract2gene.data import model_path
from webapp import auth, database, query, ui

model: a2g.model.Model | None = None
min_year = 0


async def get_client() -> AsyncQdrantClient:
    global min_year
    client = database.connect()
    if not await client.collection_exists(cfg.collection_name):
        raise RuntimeError(
            "qdrant collection not found. This might be due to changing the"
            + " model after building the collection. Rerun the populate db"
            + "script."
        )

    min_year = await query.get_min_year(client, cfg.collection_name)

    return client


def get_model() -> a2g.model.Model:
    global model
    if model is None:
        model = a2g.model.load_from_disk(cfg.model_name)

    return model


ClientDep = Annotated[AsyncQdrantClient, Depends(get_client)]
ModelDep = Annotated[a2g.model.Model, Depends(get_model)]


def _parse_years(year_min: int, year_max: int) -> tuple[int, int]:
    """Determine the range of years to filter.

    If year min and year max are both > 0, they are treated like specific
    years.
    If year min is 0, don't filter, return all publications.
    If year min is negative, treat as relative to the current year. I.e. the
    range should be from current year - min year to current year.
    """
    if year_min <= 0:
        year_max = datetime.today().year

    if year_min == 0:
        year_min = min_year

    if year_min < 0:
        year_min = year_max - abs(year_min)

    return (year_min, year_max)


with open(os.path.join(model_path(cfg.model_name), "genes.json"), "r") as js:
    genes = json.load(js)

main_router = APIRouter(dependencies=[Depends(auth.is_authorized)])


@main_router.get("/", response_class=HTMLResponse)
async def abstract2gene(request: Request, client: ClientDep):
    collection = await client.get_collection(cfg.collection_name)
    n_points = collection.points_count

    return ui.home(request, min_year, n_points)


@main_router.get("/pmid_search", response_class=HTMLResponse)
async def pmid_search_page(request: Request, client: ClientDep):
    collection = await client.get_collection(cfg.collection_name)
    n_points = collection.points_count

    return ui.pmid_search_page(request, min_year, n_points)


@main_router.post("/results/user_input", name="search")
async def post_abstract_search(
    request: Request,
    client: ClientDep,
    model: ModelDep,
    title: str = Form(...),
    abstract: str = Form(...),
    year_min: int = Form(...),
    year_max: int = Form(...),
):
    if not await client.collection_exists(cfg.tmp_collection_name):
        await database.init_db(client, model, cfg.tmp_collection_name)

    session_id = uuid.uuid4().hex
    await database.store_user_abstracts(
        client, model, title, abstract, session_id, cfg.tmp_collection_name
    )

    # May add user input for these on search page later.
    molecular = True
    behavioral = True

    base_url = "/results/user_input"
    return RedirectResponse(
        base_url
        + f"/{session_id}?year_min={year_min}&year_max={year_max}"
        + f"&page=1&molecular={molecular}&behavioral={behavioral}",
        status_code=302,
    )


@main_router.get("/results/user_input/{session_id}", name="user_input")
async def get_abstract_search(
    request: Request,
    client: ClientDep,
    session_id: str,
    year_min: int = min_year,
    year_max: int = datetime.today().year,
    page: int = 1,
    behavioral: bool = True,
    molecular: bool = True,
):
    year_rng = _parse_years(year_min, year_max)
    return await ui.results(
        request,
        client,
        session_id,
        year_rng,
        behavioral,
        molecular,
        page,
        genes,
        cfg.collection_name,
    )


@main_router.get("/results/pmid_search", name="pmid_search")
async def get_pmid_search(
    request: Request,
    client: ClientDep,
    positives: str,
    negatives: str = "",
    year_min: int = min_year,
    year_max: int = datetime.today().year,
    page: int = 1,
    behavioral: bool = True,
    molecular: bool = True,
):
    year_rng = _parse_years(year_min, year_max)

    if not positives:
        return "Must provide at least one positive example."

    positive_list = [int(pmid) for pmid in positives.split(",")]

    negative_list = []
    if negatives:
        negative_list = [int(pmid) for pmid in negatives.split(",")]

    return await ui.search_pmid(
        request,
        client,
        positive_list,
        negative_list,
        year_rng,
        behavioral,
        molecular,
        page,
        genes,
        cfg.collection_name,
    )


@main_router.get("/analyze/{pmid}")
async def analyze_references(
    request: Request,
    client: ClientDep,
    pmid: int,
    behavioral: bool = True,
    molecular: bool = True,
):
    return await ui.analyze_references(
        request,
        client,
        pmid,
        behavioral,
        molecular,
        genes,
        cfg.collection_name,
    )


app = FastAPI()
app.include_router(main_router)
app.include_router(auth.router)
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")


@app.on_event("shutdown")
async def shutdown_event():
    client = await get_client()

    if await client.collection_exists(cfg.tmp_collection_name):
        await client.delete_collection(cfg.tmp_collection_name)
