"""
RAG 运行时与流式 LLM 适配层。

本模块复用阶段 1 的切片、BM25 和 FAISS 能力，但索引来源改为 SQLite 中
所有状态为 ready 的制度文档。上传或删除知识文件后，统一重建内存索引。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import AsyncIterator, Protocol, Sequence

import httpx

try:
    from . import stage1_rag_cli as rag
    from .db import SQLiteDatabase
except ImportError:
    import stage1_rag_cli as rag
    from db import SQLiteDatabase


@dataclass(frozen=True)
class Citation:
    """返回给前端并保存到消息历史的一条制度来源。"""

    document_id: str
    file_name: str
    document_version: str
    chunk_id: int
    chunk_index: int
    excerpt: str
    score: float

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "document_version": self.document_version,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "excerpt": self.excerpt,
            "score": round(self.score, 6),
        }


class StreamingLLMProvider(Protocol):
    """流式模型供应商必须实现的统一接口。"""

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """逐段产出回答文本。"""


def clean_document_excerpt(text: str, max_length: int = 300) -> str:
    """
    整理用于回答和来源卡片展示的制度片段。

    500 字切片为了保留重叠上下文，可能从上一段半句话开始。如果前 160 字内
    出现新的二级标题，则从该标题开始展示，并移除 Markdown 标题符号。
    """

    compact = re.sub(r"\s+", " ", text).strip()
    heading_position = compact.find("## ")
    if 0 < heading_position < 160:
        compact = compact[heading_position + 3 :]
    compact = re.sub(r"\s*#{1,6}\s*", " ", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact[:max_length]


class MockStreamingLLMProvider:
    """
    无需 API Key 的离线流式模型。

    它从提示词中的检索片段提取摘要，用于验证 Web、SSE、数据库和来源展示链路。
    它不是生产回答模型，输出会明确标记为离线演示。
    """

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        user_message = messages[-1]["content"] if messages else ""
        blocks = re.findall(
            r"\[资料(\d+)\][^\n]*\n(.+?)(?=\n\n\[资料|\n\n【用户问题】|\Z)",
            user_message,
            flags=re.S,
        )
        question_match = re.search(
            r"【用户问题】\s*(.+?)(?=\n\n请回答当前问题|\Z)",
            user_message,
            flags=re.S,
        )
        question = question_match.group(1).strip() if question_match else "当前问题"

        if blocks:
            lines = [f"根据当前制度资料，关于“{question}”，可确认："]
            for number, snippet in blocks[:3]:
                lines.append(f"【资料{number}】{clean_document_excerpt(snippet, 220)}")
            lines.append("以上为离线模型演示；涉及个人待遇或制度冲突时请转人工确认。")
        else:
            lines = [
                "制度中未找到足够依据。",
                "建议补充制度文件或创建工单转人工确认。",
            ]

        full_text = "\n".join(lines)
        for index in range(0, len(full_text), 18):
            yield full_text[index : index + 18]
            await asyncio.sleep(0.015)


class OpenAICompatibleStreamingProvider:
    """兼容 OpenAI `/chat/completions` SSE 协议的供应商适配器。"""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        timeout = httpx.Timeout(
            connect=min(10.0, self.timeout),
            read=self.timeout,
            write=30.0,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self.api_url,
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"模型接口返回 HTTP {response.status_code}: {body[:500]}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content")
                    if isinstance(content, str) and content:
                        yield content


class RagRuntime:
    """
    管理多文档检索索引。

    当前请求量较小，使用进程内 FAISS IndexFlatIP。知识文件变化后通过
    `reload_from_database()` 原子替换索引。后续可替换为 pgvector 或独立服务。
    """

    def __init__(self, *, cache_dir: Path):
        self.cache_dir = cache_dir
        self.settings = rag.Settings.from_env(
            document_path=Path("unused.md"),
            top_k=3,
            cache_dir=cache_dir,
        )
        if (
            self.settings.embedding_provider == "hash"
            and os.getenv("VECTOR_WEIGHT") is None
            and os.getenv("BM25_WEIGHT") is None
        ):
            # hash 向量只用于离线测试，提高 BM25 占比使精确制度词优先。
            self.settings = replace(
                self.settings,
                vector_weight=0.20,
                bm25_weight=0.80,
            )

        self.index: rag.HybridRagIndex | None = None
        self.embedding_provider: rag.EmbeddingProvider | None = None
        self.ready = False
        self.error_message: str | None = None
        self.document_count = 0
        self.chunk_count = 0
        self._reload_lock = asyncio.Lock()

    async def reload_from_database(self, database: SQLiteDatabase) -> None:
        """
        读取所有 ready 文档切片并重建混合索引。

        构建在锁内完成，避免上传和删除并发时短暂出现两套索引互相覆盖。
        """

        async with self._reload_lock:
            rows = await database.get_ready_chunks()
            if not rows:
                self.index = None
                self.ready = False
                self.document_count = 0
                self.chunk_count = 0
                self.error_message = "暂无状态为 ready 的制度文档"
                return

            chunks = [
                rag.Chunk(
                    chunk_id=int(row["vector_id"]),
                    text=str(row["content"]),
                    start_char=int(row["char_start"]),
                    end_char=int(row["char_end"]),
                )
                for row in rows
            ]
            embedding_provider = rag.build_embedding_provider(self.settings)
            new_index = await asyncio.to_thread(
                rag.HybridRagIndex.build,
                chunks,
                embedding_provider,
                vector_weight=self.settings.vector_weight,
                bm25_weight=self.settings.bm25_weight,
            )
            self.index = new_index
            self.embedding_provider = embedding_provider
            self.document_count = len({str(row["document_id"]) for row in rows})
            self.chunk_count = len(rows)
            self.ready = True
            self.error_message = None

    async def retrieve(
        self,
        question: str,
        database: SQLiteDatabase,
    ) -> tuple[list[rag.SearchHit], list[Citation]]:
        """执行 Top-K 混合检索并映射出跨文档来源。"""

        if not self.ready or self.index is None:
            raise RuntimeError(self.error_message or "知识库尚未准备完成")
        hits = await asyncio.to_thread(
            self.index.search,
            question,
            top_k=self.settings.top_k,
        )
        if not hits:
            return [], []

        source_rows = await database.get_source_rows_by_vector_ids(
            vector_ids=[hit.chunk.chunk_id for hit in hits]
        )
        citations: list[Citation] = []
        for hit in hits:
            row = source_rows.get(hit.chunk.chunk_id)
            if row is None:
                continue
            citations.append(
                Citation(
                    document_id=str(row["document_id"]),
                    file_name=str(row["file_name"]),
                    document_version=str(row["version"]),
                    chunk_id=int(row["vector_id"]),
                    chunk_index=int(row["chunk_index"]),
                    excerpt=clean_document_excerpt(str(row["content"]), 300),
                    score=hit.hybrid_score,
                )
            )
        return hits, citations

    def build_messages(
        self,
        *,
        question: str,
        hits: Sequence[rag.SearchHit],
        history: Sequence[dict[str, str]],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """
        构建模型消息。

        system_prompt 在每次请求时从数据库传入，不在本模块写死模板。
        历史用于理解上下文，但事实依据限定为当前检索片段。
        """

        context_parts = [
            f"[资料{number}] 切片ID={hit.chunk.chunk_id}\n{hit.chunk.text}"
            for number, hit in enumerate(hits, start=1)
        ]
        context = "\n\n".join(context_parts) if context_parts else "无可用制度资料。"
        history_text = "\n".join(
            f"{'员工' if item['role'] == 'user' else '助手'}：{item['content']}"
            for item in history
        ) or "无历史对话。"
        user_prompt = (
            "【最近对话】\n"
            f"{history_text}\n\n"
            "【知识片段】\n"
            f"{context}\n\n"
            "【用户问题】\n"
            f"{question}\n\n"
            "请回答当前问题，不要在答案末尾重复知识片段全文。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def create_llm_provider(self) -> StreamingLLMProvider:
        """根据环境变量创建可替换模型供应商。"""

        provider = self.settings.llm_provider
        if provider == "mock":
            return MockStreamingLLMProvider()
        if provider == "openai_compatible":
            if not self.settings.llm_api_url:
                raise RuntimeError("缺少 LLM_API_URL")
            if not self.settings.llm_api_key:
                raise RuntimeError("缺少 LLM_API_KEY")
            if not self.settings.llm_model:
                raise RuntimeError("缺少 LLM_MODEL")
            return OpenAICompatibleStreamingProvider(
                api_url=self.settings.llm_api_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                timeout=self.settings.request_timeout,
            )
        raise RuntimeError(f"不支持的 LLM_PROVIDER: {provider}")

    async def stream_answer(
        self,
        *,
        question: str,
        hits: Sequence[rag.SearchHit],
        history: Sequence[dict[str, str]],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """组合 Prompt 和流式模型输出。"""

        provider = self.create_llm_provider()
        messages = self.build_messages(
            question=question,
            hits=hits,
            history=history,
            system_prompt=system_prompt,
        )
        async for delta in provider.stream(messages):
            yield delta
