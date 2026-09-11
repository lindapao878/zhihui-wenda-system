"""Knowledge query LangGraph definition."""
from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_process.config import normalize_retrieval_mode, get_config
from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.hyde_search_node import HyDeSearchNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.knowledge_graph_node import KnowledgeGraphQueryNode
from knowledge.processor.query_process.nodes.mcp_search_node import McpSearchNode
from knowledge.processor.query_process.nodes.rerank_node import RerankNode
from knowledge.processor.query_process.nodes.rrf_node import RrfNode
from knowledge.processor.query_process.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_process.state import QueryGraphState


def route_after_item_confirm(state: QueryGraphState) -> bool:
    return bool(state.get("answer"))


def route_after_vector(state: QueryGraphState) -> str:
    """Auto mode: extend only when plain vector recall found nothing usable."""
    return "direct" if state.get("embedding_chunks") else "extended"


def create_query_graph(retrieval_mode: Optional[str] = None) -> CompiledStateGraph:
    mode = normalize_retrieval_mode(retrieval_mode or get_config().retrieval_mode)
    workflow = StateGraph(QueryGraphState)

    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "search_embedding": VectorSearchNode(),
        "join": lambda x: {},
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode(),
    }
    if mode in ("full", "hyde", "auto"):
        nodes["search_embedding_hyde"] = HyDeSearchNode()
    if mode in ("full", "auto"):
        nodes["web_search_mcp"] = McpSearchNode()
        nodes["knowledge_graph_query"] = KnowledgeGraphQueryNode()
    if mode in ("full", "hyde"):
        nodes["multi_search"] = lambda x: {}
    if mode == "auto":
        nodes["multi_search_extended"] = lambda x: {}

    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("item_name_confirm")
    retrieval_entry = "multi_search" if mode in ("full", "hyde") else "search_embedding"
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {False: retrieval_entry, True: "answer_output"},
    )

    if mode in ("full", "hyde"):
        workflow.add_edge("multi_search", "search_embedding")
        workflow.add_edge("multi_search", "search_embedding_hyde")
        workflow.add_edge("search_embedding", "join")
        workflow.add_edge("search_embedding_hyde", "join")
        if mode == "full":
            workflow.add_edge("multi_search", "web_search_mcp")
            workflow.add_edge("web_search_mcp", "join")
        workflow.add_edge("join", "rrf")
        workflow.add_edge("rrf", "rerank")
        if mode == "full":
            workflow.add_edge("rerank", "knowledge_graph_query")
            workflow.add_edge("knowledge_graph_query", "answer_output")
        else:
            workflow.add_edge("rerank", "answer_output")
    elif mode == "basic":
        workflow.add_edge("search_embedding", "join")
        workflow.add_edge("join", "rrf")
        workflow.add_edge("rrf", "rerank")
        workflow.add_edge("rerank", "answer_output")
    elif mode == "auto":
        workflow.add_conditional_edges(
            "search_embedding",
            route_after_vector,
            {"direct": "join", "extended": "multi_search_extended"},
        )
        workflow.add_edge("multi_search_extended", "search_embedding_hyde")
        workflow.add_edge("multi_search_extended", "web_search_mcp")
        workflow.add_edge("search_embedding_hyde", "join")
        workflow.add_edge("web_search_mcp", "join")
        workflow.add_edge("join", "rrf")
        workflow.add_edge("rrf", "rerank")
        workflow.add_edge("rerank", "knowledge_graph_query")
        workflow.add_edge("knowledge_graph_query", "answer_output")

    workflow.add_edge("answer_output", END)

    return workflow.compile()


query_app = create_query_graph()
