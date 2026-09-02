"""Document import LangGraph definition."""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.docx_to_md_node import DocxToMdNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.entity_recognition_node import EntityRecognitionNode
from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img_node import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def import_router(state: ImportGraphState) -> str:
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_docx_read_enabled"):
        return "docx_to_md_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return END


def create_import_graph() -> CompiledStateGraph:
    graph_pipeline = StateGraph(ImportGraphState)

    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "docx_to_md_node": DocxToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "entity_rec_node": EntityRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    graph_pipeline.set_entry_point("entry_node")
    for key, value in nodes.items():
        graph_pipeline.add_node(key, value)

    graph_pipeline.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "md_img_node": "md_img_node",
            "pdf_to_md_node": "pdf_to_md_node",
            "docx_to_md_node": "docx_to_md_node",
            END: END,
        },
    )
    graph_pipeline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pipeline.add_edge("docx_to_md_node", "md_img_node")
    graph_pipeline.add_edge("md_img_node", "document_split_node")
    graph_pipeline.add_edge("document_split_node", "item_name_rec_node")
    graph_pipeline.add_edge("item_name_rec_node", "entity_rec_node")
    graph_pipeline.add_edge("entity_rec_node", "bge_embedding_node")
    graph_pipeline.add_edge("bge_embedding_node", "import_milvus_node")
    graph_pipeline.add_edge("import_milvus_node", END)

    return graph_pipeline.compile()


kb_import_graph_app = create_import_graph()


def run_import_graph(import_file_path: str, file_dir: str) -> dict:
    state = {"import_file_path": import_file_path, "file_dir": file_dir}
    init_state = create_default_state(**state)
    final_state = None
    for event in kb_import_graph_app.stream(init_state):
        for _node_name, node_state in event.items():
            final_state = node_state
    return final_state
