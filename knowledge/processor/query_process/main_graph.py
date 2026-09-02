"""Knowledge query LangGraph definition."""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

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


def create_query_graph() -> CompiledStateGraph:
    workflow = StateGraph(QueryGraphState)

    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: {},
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "web_search_mcp": McpSearchNode(),
        "join": lambda x: {},
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "knowledge_graph_query": KnowledgeGraphQueryNode(),
        "answer_output": AnswerOutputNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("item_name_confirm")
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {False: "multi_search", True: "answer_output"},
    )

    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("web_search_mcp", "join")

    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "knowledge_graph_query")
    workflow.add_edge("knowledge_graph_query", "answer_output")
    workflow.add_edge("answer_output", END)

    return workflow.compile()


query_app = create_query_graph()

