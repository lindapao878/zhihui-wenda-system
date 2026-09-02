"""Knowledge query graph state."""
from __future__ import annotations

import copy
from typing import Annotated, List, TypedDict
from operator import add


class QueryGraphState(TypedDict, total=False):
    session_id: str
    task_id: str
    message_id: str
    original_query: str
    embedding_chunks: list
    hyde_embedding_chunks: list
    rrf_chunks: list
    web_search_docs: list
    reranked_docs: list
    prompt: str
    answer: str
    item_names: List[str]
    related_entities: List[str]
    rewritten_query: str
    history: list
    kg_triples: list
    is_stream: bool


DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "related_entities": [],
    "rewritten_query": "",
    "history": [],
    "kg_triples": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    return copy.deepcopy(DEFAULT_STATE)


graph_default_state = DEFAULT_STATE

