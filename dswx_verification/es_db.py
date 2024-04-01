from functools import lru_cache
import warnings

import urllib3
import pandas as pd
from dotenv import dotenv_values
from elasticsearch import Elasticsearch, client
from elasticsearch_dsl import Q, Search

urllib3.disable_warnings()
INDICES = {"dswx_hls": "grq_*_hls-2023.09",
           "dswx_s1": "grq_v0.1_l3_dswx_s1-*",
           "rtc": "grq_v*_l2_rtc_s1-2024.03"}


@lru_cache
def get_es_search_client(prod="dswx_hls") -> client.Elasticsearch:
    if prod not in INDICES.keys():
        raise ValueError(f'prod must be {', '.join(INDICES.keys())}')
    index = INDICES[prod]

    config = dotenv_values()
    ES_USERNAME = config["ES_USERNAME"]
    ES_PASSWORD = config["ES_PASSWORD"]
    GRQ_URL = "https://100.104.62.10/grq_es/"
    grq_client = Elasticsearch(
        GRQ_URL,
        http_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=False,
        read_timeout=50000,
        terminate_after=2500,
    )
    search = Search(using=grq_client, index=index)

    if not grq_client.ping():
        raise ValueError("Either JPL username/password is wrong or not connected to VPN")

    return search


def get_dswx_hls_doc(hls_id: str) -> dict:
    """Raises error if no HLS ID found"""
    search = get_es_search_client(prod="dswx_hls")
    q_qs = Q("query_string", query=f'"{hls_id}"', default_field="metadata.input_granule_id")

    query = search.query(q_qs)
    resp = query.execute()
    n = len(resp.hits)

    if n > 1:
        raise ValueError("Multiple DSWx Products for current query")
    if n == 0:
        raise ValueError("No DSWx products match HLS ID")

    data = resp.hits[0].to_dict()
    return data


def get_dswx_hls_urls(hls_id: str) -> list:
    base_url = "https://opera-pst-rs-pop1.s3.us-west-2.amazonaws.com/"
    doc = get_dswx_hls_doc(hls_id)
    paths = doc["metadata"]["product_s3_paths"]
    paths_formatted = [path.replace("s3://opera-pst-rs-pop1/", "") for path in paths]
    urls = [f"{base_url}{path}" for path in paths_formatted]
    return urls


def get_dswx_s1_doc(mgrs_tile: str, dt: pd.Timestamp | str) -> dict:
    dt = pd.to_datetime(dt)
    search = get_es_search_client(prod="dswx_s1")

    date_token = f"{dt.year}{dt.month:02d}{dt.day:02d}"
    mgrs_token = mgrs_tile

    query = Q("simple_query_string", query=f"{date_token} {mgrs_token}", fields=["id"], default_operator="and")

    query_ob = search.query(query)
    resp = query_ob.execute()
    hits = [hit.to_dict() for hit in resp.hits]
    if len(hits) > 1:
        warnings.warn(f"Multiple hits found for {mgrs_tile} and {date_token}")
    return hits


def get_dswx_s1_docs_in_date_range(mgrs_tile: str, dt: pd.Timestamp | str, buffer_days=7) -> list[dict]:
    """Orders docs by proximity to provided datetime"""
    dt = pd.to_datetime(dt)
    dates = [dt + pd.Timedelta(days=n_days) for n_days in range(-buffer_days, buffer_days + 1)]
    doc_pairs = [(dt_, doc) for dt_ in dates for doc in get_dswx_s1_doc(mgrs_tile, dt_) if doc]
    doc_pairs_ordered = sorted(doc_pairs, key=lambda pair: abs((pair[0] - dt).days))
    docs = []
    if doc_pairs_ordered:
        _, docs = zip(*doc_pairs_ordered)
        docs = list(docs)
    else:
        warnings.warn(f'No docs found, check the query for {mgrs_tile} and {dt}!', category=UserWarning)
    return docs


def get_rtc_doc(rtc_id: str):
    search = get_es_search_client(prod="rtc")
    query = search.query("match", _id=rtc_id)
    response = query.execute()
    hits = response.hits
    n_hits = len(hits)
    if n_hits != 1:
        raise ValueError(f'Expecting exactly one hit, but got {n_hits}')
    return hits[0].to_dict()
