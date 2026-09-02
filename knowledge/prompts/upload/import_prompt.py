"""Import prompt templates loaded from .prompt files."""
from knowledge.prompts.loader import load_prompt

ITEM_NAME_SYSTEM_PROMPT = load_prompt("upload", "import_item_name_system.prompt")
ITEM_NAME_USER_PROMPT_TEMPLATE = load_prompt("upload", "import_item_name.prompt")
