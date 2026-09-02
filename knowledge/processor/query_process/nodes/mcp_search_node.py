"""MCP web search node."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.logger_util import logger



class McpSearchNode(BaseNode):
    name = "mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_query, _validated_item_names = self._validate_query_inputs(state)
        if not self.config.mcp_dashscope_base_url:
            logger.info("未配置 MCP_DASHSCOPE_BASE_URL，跳过网络搜索")
            return {}

        mcp_result = self._run_async(self._create_execute_web_search(validated_query))
        if not mcp_result:
            return {}
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state):
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(self.name, "rewritten_query", str)
        if not isinstance(item_names, list):
            raise StateFieldError(self.name, "item_names", list)
        return rewritten_query, item_names

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    async def _create_execute_web_search(self, query: str) -> List[Dict[str, Any]]:
        try:
            from agents.mcp import MCPServerStreamableHttp
        except Exception as exc:
            logger.warning("agents.mcp 不可用: {}", exc)
            return []

        api_key = self.config.mcp_dashscope_api_key or self.config.openai_api_key

        mcp_client = MCPServerStreamableHttp(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {api_key}"},
                "timeout": 300,
                "sse_read_timeout": 300,
            },
        )

        try:
            await mcp_client.connect()
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 3},
            )
            if not execute_tool_result or not execute_tool_result.content[0]:
                return []

            text_content = execute_tool_result.content[0].text
            if not text_content:
                return []

            payload = json.loads(text_content)
            pages = payload.get("pages", [])
            search_result = []
            for page in pages:
                search_result.append({
                    "snippet": page.get("snippet", "").strip(),
                    "title": page.get("title", "").strip(),
                    "url": page.get("url", "").strip(),
                })
            logger.info("MCP 网络搜索完成, 网页文档数={}", len(search_result))
            return search_result
        except Exception as exc:
            logger.warning("网络搜索失败: {}", exc)
            return []
        finally:
            try:
                await mcp_client.cleanup()
            except Exception:
                pass
