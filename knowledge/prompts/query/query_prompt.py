"""Query prompt templates loaded from .prompt files."""
from knowledge.prompts.loader import load_prompt

ITEM_NAME_EXTRACT_TEMPLATE = load_prompt("query", "item_name_extract.prompt")
USER_HYDE_PROMPT_TEMPLATE = load_prompt("query", "hyde.prompt")
ANSWER_PROMPT = load_prompt("query", "answer.prompt")
KG_EXTRACT_SYSTEM_PROMPT = load_prompt("query", "kg_extract_system.prompt")
KG_EXTRACT_USER_PROMPT_TEMPLATE = load_prompt("query", "kg_extract.prompt")
