"""Entry point for the search engine's web UI."""

import json
import os
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import abstract2gene as a2g
import webapp.config as cfg
from abstract2gene.data import model_path
from webapp import auth, database, query, ui

model: a2g.model.Model | None = None


def get_model() -> a2g.model.Model:
    global model
    if model is None:
        model = a2g.model.load_from_disk(cfg.model_name)

    return model


ModelDep = Annotated[a2g.model.Model, Depends(get_model)]

client = database.connect()
min_year = query.get_min_year(client, cfg.collection_name)

with open(os.path.join(model_path(cfg.model_name), "genes.json"), "r") as js:
    genes = json.load(js)

main_router = APIRouter(dependencies=[Depends(auth.is_authorized)])
main_router.mount(
    "/static", StaticFiles(directory="webapp/static"), name="static"
)

if not client.collection_exists(cfg.collection_name):
    raise RuntimeError(
        "qdrant collection not found. This might be due to changing the"
        + " model after building the collection. Rerun the populate db script."
    )


@main_router.get("/", response_class=HTMLResponse)
def abstract2gene(request: Request):
    return ui.home(request, min_year)


@main_router.get("/pmid_search", response_class=HTMLResponse)
def pmid_search_page(request: Request):
    return ui.pmid_search_page(request, min_year)


@main_router.post("/results/user_input", name="search")
def post_abstract_search(
    request: Request,
    model: ModelDep,
    title: str = Form(...),
    abstract: str = Form(...),
    year_min: int = Form(...),
    year_max: int = Form(...),
):
    if not client.collection_exists(cfg.tmp_collection_name):
        database.init_db(client, model, cfg.tmp_collection_name)

    session_id = uuid.uuid4().hex
    database.store_user_abstracts(
        client, model, title, abstract, session_id, cfg.tmp_collection_name
    )

    return RedirectResponse(
        f"/results/user_input/{session_id}?year_min={year_min}&year_max={year_max}&page=1",
        status_code=302,
    )


@main_router.get("/results/user_input/{session_id}", name="user_input")
def get_abstract_search(
    request: Request,
    session_id: str,
    year_min: int = min_year,
    year_max: int = datetime.today().year,
    page: int = 1,
):
    return ui.results(
        request,
        client,
        session_id,
        (year_min, year_max),
        page,
        genes,
        cfg.collection_name,
    )


@main_router.get("/results/pmid_search", name="pmid_search")
def get_pmid_search(
    request: Request,
    positives: str,
    negatives: str = "",
    year_min: int = min_year,
    year_max: int = datetime.today().year,
    page: int = 1,
):

    if not positives:
        return "Must provide at least one positive example."

    positive_list = [int(pmid) for pmid in positives.split(",")]

    negative_list = []
    if negatives:
        negative_list = [int(pmid) for pmid in negatives.split(",")]

    return ui.search_pmid(
        request,
        client,
        positive_list,
        negative_list,
        (year_min, year_max),
        page,
        genes,
        cfg.collection_name,
    )


@main_router.get("/analyze/{pmid}")
def analyze_references(request: Request, pmid: int):
    return ui.analyze_references(
        request, client, pmid, genes, cfg.collection_name
    )


app = FastAPI()
app.include_router(main_router)
app.include_router(auth.router)


@app.on_event("shutdown")
def shutdown_event():
    if client.collection_exists(cfg.tmp_collection_name):
        client.delete_collection(cfg.tmp_collection_name)
