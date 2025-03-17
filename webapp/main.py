"""Entry point for the search engine's web UI."""

import json
import os
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import abstract2gene as a2g
import webapp.config as cfg
from abstract2gene.data import model_path
from webapp import database, query, ui

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

app = FastAPI()
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")


if not client.collection_exists(cfg.collection_name):
    raise RuntimeError(
        "qdrant collection not found. This might be due to changing the"
        + " model after building the collection. Rerun the populate db script."
    )


@app.get("/", response_class=HTMLResponse)
def abstract2gene(request: Request):
    return ui.home(request, min_year)


@app.get("/pmid_search", response_class=HTMLResponse)
def pmid_search_page(request: Request):
    return ui.pmid_search_page(request, min_year)


@app.post("/results/user_input", name="search")
def post_abstract_search(
    request: Request,
    model: ModelDep,
    title: str = Form(...),
    abstract: str = Form(...),
    year_min: int = Form(...),
    year_max: int = Form(...),
):
    return ui.results(
        request,
        client,
        model,
        title,
        abstract,
        (year_min, year_max),
        genes,
        cfg.collection_name,
    )


@app.get("/results/pmid_search")
def post_pmid_search(
    request: Request,
    positives: str,
    negatives: str = "",
    year_min: int = min_year,
    year_max: int = datetime.today().year,
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
        genes,
        cfg.collection_name,
    )


@app.get("/analyze/{pmid}")
def analyze_references(request: Request, pmid: int):
    return ui.analyze_references(
        request, client, pmid, genes, cfg.collection_name
    )
