#!/usr/bin/env python3
"""
员工薪酬福利咨询工单分流系统 - 阶段 1：本地 RAG 命令行原型

这个脚本只验证最小的“文档入库 -> 混合检索 -> 上下文增强 -> LLM 回答”链路，
不包含 Web、数据库、用户登录、工单持久化和多轮对话持久化。

依赖安装：
    pip install faiss-cpu numpy

运行示例：
    python stage1_rag_cli.py --document ./docs/薪酬福利制度.md
    python stage1_rag_cli.py --document ./docs/报销制度.md --question "差旅报销需要哪些材料？"

可配置环境变量：
    LLM_PROVIDER:
        - mock（默认，在未配置 API Key 时用于离线验证）
        - openai_compatible（任何兼容 Chat Completions 的供应商）
    LLM_API_URL:
        - 完整 Chat Completions 地址，例如 https://api.openai.com/v1/chat/completions
    LLM_API_KEY:
        - LLM 服务的 API Key
    LLM_MODEL:
        - 模型名称
    EMBEDDING_PROVIDER:
        - hash（默认，离线测试用的确定性哈希向量）
        - remote（OpenAI-Compatible Embeddings API）
    EMBEDDING_API_URL:
        - 完整 Embeddings 地址，例如 https://api.openai.com/v1/embeddings
    EMBEDDING_API_KEY:
        - 向量模型 API Key；未填写时 EMBEDDING_PROVIDER 应使用 hash
    EMBEDDING_MODEL:
        - 向量模型名称，例如 text-embedding-3-small
    EMBEDDING_DIMENSION:
        - hash 向量维度，默认 384
    CHUNK_SIZE=500
    CHUNK_OVERLAP=80
    TOP_K=3

说明：
1. 文档按 500 个字符切片，并保留 80 字符重叠，避免语义被硬切断。
2. hash 向量只用于离线跑通链路，不具备真实语义检索能力；生产环境应使用 remote 向量模型。
3. 更换文档、切片参数或向量模型后，索引缓存会自动失效并重建。
4. 发送到外部模型前，当前原型只做基础敏感信息提示，不替代正式脱敏模块。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 【核心模块｜后续 FastAPI 阶段直接复用】
# FAISS 作为向量索引实现。生产环境也可能替换为 Milvus、pgvector 或 Elasticsearch，
# 因此后续应把 VectorIndex 抽象保留在独立模块中；本阶段先集中在单文件脚本里。
# ---------------------------------------------------------------------------
try:
    import faiss
except ImportError as exc:  # pragma: no cover - 环境缺少依赖时给出明确提示
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:
    FAISS_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCUMENT = BASE_DIR / "docs" / "薪酬福利制度.md"
DEFAULT_CACHE_DIR = BASE_DIR / ".rag_cache"


@dataclass(frozen=True)
class Settings:
    """
    【核心配置对象】

    所有运行参数集中在这里，避免后续迁移到 FastAPI 时环境变量散落在业务代码中。
    本阶段使用环境变量和命令行参数；后续可改为 Pydantic Settings。
    """

    document_path: Path
    cache_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    vector_weight: float
    bm25_weight: float
    embedding_provider: str
    embedding_api_url: str | None
    embedding_api_key: str | None
    embedding_model: str | None
    embedding_dimension: int
    llm_provider: str
    llm_api_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    request_timeout: float

    @classmethod
    def from_env(
        cls,
        *,
        document_path: Path | None = None,
        top_k: int | None = None,
        cache_dir: Path | None = None,
    ) -> "Settings":
        """
        从环境变量生成配置，并允许命令行参数覆盖文档路径、Top-K 和缓存目录。

        默认行为：
        - 未提供 LLM Key 时使用 mock 模型，保证脚本可以离线演示完整检索链路。
        - 未提供 Embedding Key 时使用 hash 向量，保证 FAISS 链路可立即验证。
        """

        chunk_size = _env_int("CHUNK_SIZE", 500)
        chunk_overlap = _env_int("CHUNK_OVERLAP", 80)
        vector_weight = _env_float("VECTOR_WEIGHT", 0.65)
        bm25_weight = _env_float("BM25_WEIGHT", 0.35)

        if chunk_size <= 0:
            raise ValueError("CHUNK_SIZE 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE")
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("检索权重不能为负数")
        if vector_weight + bm25_weight == 0:
            raise ValueError("向量权重和 BM25 权重不能同时为 0")

        weight_sum = vector_weight + bm25_weight

        embedding_api_key = os.getenv("EMBEDDING_API_KEY") or None
        embedding_provider = os.getenv(
            "EMBEDDING_PROVIDER",
            "remote" if embedding_api_key else "hash",
        ).strip().lower()

        llm_api_key = os.getenv("LLM_API_KEY") or None
        llm_provider = os.getenv(
            "LLM_PROVIDER",
            "openai_compatible" if llm_api_key else "mock",
        ).strip().lower()

        return cls(
            document_path=document_path or Path(
                os.getenv("DOCUMENT_PATH", str(DEFAULT_DOCUMENT))
            ),
            cache_dir=cache_dir or Path(os.getenv("RAG_CACHE_DIR", str(DEFAULT_CACHE_DIR))),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k or _env_int("TOP_K", 3),
            vector_weight=vector_weight / weight_sum,
            bm25_weight=bm25_weight / weight_sum,
            embedding_provider=embedding_provider,
            embedding_api_url=os.getenv("EMBEDDING_API_URL") or None,
            embedding_api_key=embedding_api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL") or None,
            embedding_dimension=_env_int("EMBEDDING_DIMENSION", 384),
            llm_provider=llm_provider,
            llm_api_url=os.getenv("LLM_API_URL") or None,
            llm_api_key=llm_api_key,
            llm_model=os.getenv("LLM_MODEL") or None,
            request_timeout=_env_float("REQUEST_TIMEOUT", 60.0),
        )


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；非法值直接报错，避免静默使用错误配置。"""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值为 {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    """读取浮点数环境变量；非法值直接报错。"""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字，当前值为 {value!r}") from exc


@dataclass(frozen=True)
class Chunk:
    """
    【核心数据结构｜后续阶段复用】

    一个 Chunk 对应检索与引用所需的最小文档单位。
    后续入库时可以继续增加 document_id、version、department_scope 等元数据。
    """

    chunk_id: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class SearchHit:
    """混合检索的单个结果，同时保留向量分数和 BM25 分数以便解释召回依据。"""

    chunk: Chunk
    vector_score: float
    bm25_score: float
    hybrid_score: float


class EmbeddingProvider(Protocol):
    """
    【核心接口｜供应商可替换】

    文档向量化和查询向量化必须使用同一供应商、同一模型和同一维度。
    后续可以把此接口迁移到 app/rag/embedding.py。
    """

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """将文档切片批量转换为二维 float32 向量矩阵。"""

    def embed_query(self, text: str) -> np.ndarray:
        """将用户问题转换为单条 float32 向量。"""


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    对向量做 L2 归一化。

    FAISS 的 IndexFlatIP 使用内积。归一化后，内积等价于余弦相似度，
    因此向量分数会被稳定限制在常见余弦范围内。
    """

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashEmbeddingProvider:
    """
    【离线测试专用｜非生产向量模型】

    使用字符和词特征的稳定哈希生成固定维度向量。
    它不包含语义理解，只用于没有 API Key 时快速验证：
    切片 -> 向量矩阵 -> FAISS -> 混合检索 -> LLM 调用。
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _vectorize_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = tokenize_for_retrieval(text)
        if not tokens:
            return vector

        # 使用带符号哈希将同一 token 稳定映射到同一个维度，减少普通哈希碰撞的影响。
        frequencies = Counter(tokens)
        for token, frequency in frequencies.items():
            digest = hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=8,
            ).digest()
            number = int.from_bytes(digest, byteorder="big", signed=False)
            index = number % self.dimension
            sign = 1.0 if (number >> 1) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(frequency))
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _l2_normalize(
            np.vstack([self._vectorize_one(text) for text in texts])
        )

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vectorize_one(text))[0]


class OpenAICompatibleEmbeddingProvider:
    """
    【核心模块｜真实向量化】

    使用 OpenAI-Compatible /embeddings 协议。该协议可适配 OpenAI、DeepSeek、
    Qwen、企业自建网关等兼容供应商，避免业务层绑定具体 SDK。
    """

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

    def _request_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "input": list(texts),
        }
        result = _post_json(
            url=self.api_url,
            payload=payload,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        data = result.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("Embeddings API 未返回有效 data 字段")

        ordered = sorted(
            data,
            key=lambda item: int(item.get("index", 0)),
        )
        vectors = []
        for item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("Embeddings API 返回了空向量")
            vectors.append(embedding)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._request_embeddings(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._request_embeddings([text])[0]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """
    【核心工厂函数｜供应商可替换】

    业务编排层只依赖 EmbeddingProvider 接口，不直接依赖 hash 或某个厂商。
    """

    provider = settings.embedding_provider
    if provider == "hash":
        print(
            "[提示] 当前使用 hash 向量。它只适合离线验证，"
            "生产环境请配置 EMBEDDING_PROVIDER=remote。"
        )
        return HashEmbeddingProvider(settings.embedding_dimension)

    if provider == "remote":
        if not settings.embedding_api_url:
            raise ValueError("EMBEDDING_PROVIDER=remote 时必须配置 EMBEDDING_API_URL")
        if not settings.embedding_api_key:
            raise ValueError("EMBEDDING_PROVIDER=remote 时必须配置 EMBEDDING_API_KEY")
        if not settings.embedding_model:
            raise ValueError("EMBEDDING_PROVIDER=remote 时必须配置 EMBEDDING_MODEL")
        return OpenAICompatibleEmbeddingProvider(
            api_url=settings.embedding_api_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            timeout=settings.request_timeout,
        )

    raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# 【核心模块｜后续阶段复用】
# 中文检索分词、BM25 和混合排序均集中在以下函数中。
# 后续可以将它们迁移到 app/rag/retrieval.py，并保持输入输出结构不变。
# ---------------------------------------------------------------------------
CHINESE_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*")


def tokenize_for_retrieval(text: str) -> list[str]:
    """
    对中文和英文做轻量分词。

    原型的约束是“单脚本、低依赖”，因此不引入 jieba：
    - 英文和数字按连续 token 处理；
    - 中文连续片段同时生成单字与双字 token；
    - 双字 token 能覆盖“政策版本”“报销材料”等常见检索短语。
    """

    normalized = text.lower()
    tokens: list[str] = []
    tokens.extend(WORD_RE.findall(normalized))
    for run in CHINESE_RUN_RE.findall(normalized):
        tokens.extend(list(run))
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return [token for token in tokens if token.strip()]


class BM25Index:
    """
    【核心模块｜后续阶段复用】

    单脚本内实现 Okapi BM25，避免为了最小原型额外安装分词和 BM25 依赖。
    在数据量扩大后，可替换为 Elasticsearch/OpenSearch 的 BM25。
    """

    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = [list(tokens) for tokens in documents]
        self.k1 = k1
        self.b = b
        self.document_count = len(self.documents)
        self.document_lengths = [len(tokens) for tokens in self.documents]
        self.average_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

        document_frequencies: Counter[str] = Counter()
        for tokens in self.documents:
            document_frequencies.update(set(tokens))

        self.document_frequencies = document_frequencies
        self.idf = {
            token: math.log(
                1.0
                + (
                    self.document_count - frequency + 0.5
                )
                / (frequency + 0.5)
            )
            for token, frequency in document_frequencies.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """返回查询对每个文档切片的 BM25 分数。"""

        scores = np.zeros(self.document_count, dtype=np.float32)
        if not self.document_count or not query_tokens:
            return scores

        query_frequency = Counter(query_tokens)
        for index, tokens in enumerate(self.documents):
            term_frequency = Counter(tokens)
            document_length = self.document_lengths[index] or 1
            length_ratio = document_length / max(self.average_length, 1.0)

            score = 0.0
            for token, query_count in query_frequency.items():
                frequency = term_frequency.get(token, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                score += (
                    self.idf.get(token, 0.0)
                    * frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_count
                )
            scores[index] = score
        return scores


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    将一组分数缩放到 0-1。

    由于 BM25 和余弦相似度量纲不同，混合排序前必须分别归一化。
    如果全部分数相同，则返回全 0，避免错误放大无区分度的结果。
    """

    if scores.size == 0:
        return scores
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if math.isclose(minimum, maximum):
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - minimum) / (maximum - minimum)


def load_markdown(document_path: Path) -> str:
    """
    【核心模块｜后续文档入库流程复用】

    当前只处理 UTF-8 Markdown。后续阶段应在此位置增加 PDF、DOCX 解析，
    但下游始终接收统一的纯文本，避免改变切片和索引接口。
    """

    if not document_path.exists():
        raise FileNotFoundError(f"知识文件不存在: {document_path}")
    if not document_path.is_file():
        raise ValueError(f"知识路径不是文件: {document_path}")

    text = document_path.read_text(encoding="utf-8")
    # 去掉 YAML front matter，但保留正文中的 Markdown 标题和列表。
    if text.startswith("---"):
        front_matter_end = text.find("\n---", 3)
        if front_matter_end != -1:
            text = text[front_matter_end + 4 :]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("知识文件内容为空")
    return normalized


def chunk_markdown(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """
    【核心模块｜后续阶段复用】

    按字符数切分 Markdown，单块最多 chunk_size 个字符。
    为了尽量保持段落完整，结束位置会优先寻找附近的换行；找不到时再硬切。
    下一块保留 overlap 个字符，避免问题答案跨切片后失去上下文。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end

        # 在切片后半段内优先寻找段落边界；找不到时寻找普通换行。
        if hard_end < text_length:
            lower_bound = start + max(chunk_size // 2, 1)
            paragraph_end = text.rfind("\n\n", lower_bound, hard_end)
            line_end = text.rfind("\n", lower_bound, hard_end)
            if paragraph_end >= lower_bound:
                end = paragraph_end + 2
            elif line_end >= lower_bound:
                end = line_end + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=end,
                )
            )

        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


class HybridRagIndex:
    """
    【核心模块｜后续阶段复用】

    封装 FAISS 向量索引与 BM25 索引，并对外暴露统一的 build/search 接口。
    后续迁移到服务端时，可保持 retrieve(question, top_k) 的返回结构。
    """

    def __init__(
        self,
        *,
        chunks: Sequence[Chunk],
        index: object,
        bm25: BM25Index,
        embedding_provider: EmbeddingProvider,
        vector_weight: float,
        bm25_weight: float,
    ):
        self.chunks = list(chunks)
        self.index = index
        self.bm25 = bm25
        self.embedding_provider = embedding_provider
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedding_provider: EmbeddingProvider,
        *,
        vector_weight: float,
        bm25_weight: float,
    ) -> "HybridRagIndex":
        """构造 FAISS 与 BM25 双索引。"""

        if faiss is None:
            raise RuntimeError(
                "缺少 faiss 依赖。请先执行：pip install faiss-cpu"
            ) from FAISS_IMPORT_ERROR
        if not chunks:
            raise ValueError("没有可索引的文档切片")

        vectors = embedding_provider.embed_documents([chunk.text for chunk in chunks])
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RuntimeError("Embedding 数量与文档切片数量不一致")

        vectors = _l2_normalize(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))

        bm25 = BM25Index(
            [tokenize_for_retrieval(chunk.text) for chunk in chunks]
        )
        return cls(
            chunks=chunks,
            index=index,
            bm25=bm25,
            embedding_provider=embedding_provider,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

    def search(self, question: str, *, top_k: int = 3) -> list[SearchHit]:
        """
        【核心模块｜混合检索】

        1. FAISS 对问题向量执行余弦相似度召回；
        2. BM25 对关键词执行词法召回；
        3. 两种分数分别 min-max 归一化后加权融合；
        4. 返回最终分数最高的 Top-K 片段。

        混合检索能同时覆盖“同义表达”和“精确政策词”两类查询。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not self.chunks:
            return []

        query_vector = np.asarray(
            self.embedding_provider.embed_query(question),
            dtype=np.float32,
        ).reshape(1, -1)
        query_vector = _l2_normalize(query_vector)

        vector_scores = np.zeros(len(self.chunks), dtype=np.float32)
        candidate_count = min(max(top_k * 8, top_k), len(self.chunks))
        faiss_scores, faiss_indices = self.index.search(query_vector, candidate_count)
        for score, index in zip(faiss_scores[0], faiss_indices[0]):
            if index >= 0:
                vector_scores[int(index)] = float(score)

        bm25_scores = self.bm25.scores(tokenize_for_retrieval(question))
        vector_norm = _min_max_normalize(vector_scores)
        bm25_norm = _min_max_normalize(bm25_scores)
        hybrid_scores = (
            self.vector_weight * vector_norm
            + self.bm25_weight * bm25_norm
        )

        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            SearchHit(
                chunk=self.chunks[int(index)],
                vector_score=float(vector_scores[int(index)]),
                bm25_score=float(bm25_scores[int(index)]),
                hybrid_score=float(hybrid_scores[int(index)]),
            )
            for index in ranked_indices
        ]


def build_cache_namespace(settings: Settings, document_text: str) -> str:
    """
    生成缓存命名空间。

    文档内容、切片参数、向量供应商、向量模型和维度只要有一项变化，
    就使用新的索引文件，避免把旧向量与新文档混用。
    """

    fingerprint = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
    raw = "|".join(
        [
            fingerprint,
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            settings.embedding_provider,
            settings.embedding_model or "",
            str(settings.embedding_dimension),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def persist_index(
    index: HybridRagIndex,
    *,
    namespace: str,
    cache_dir: Path,
) -> None:
    """
    【后续阶段可替换为正式向量数据库】

    FAISS 索引和切片内容分别保存为文件，不保存大型 Python pickle 对象，
    以避免远程加载不可信对象带来的安全风险。
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / f"{namespace}.faiss"
    chunks_path = cache_dir / f"{namespace}.chunks.json"
    metadata_path = cache_dir / f"{namespace}.meta.json"

    faiss.write_index(index.index, str(index_path))
    chunks_payload = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        for chunk in index.chunks
    ]
    chunks_path.write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "namespace": namespace,
                "chunk_count": len(index.chunks),
                "vector_weight": index.vector_weight,
                "bm25_weight": index.bm25_weight,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_or_build_index(
    document_text: str,
    settings: Settings,
    *,
    rebuild: bool = False,
) -> HybridRagIndex:
    """
    【核心模块｜索引生命周期】

    优先加载缓存；缓存不存在、参数变化或指定 --rebuild-index 时重新构造。
    这样每次启动 CLI 不需要重复调用外部 Embeddings API。
    """

    if faiss is None:
        raise RuntimeError(
            "缺少 faiss 依赖。请先执行：pip install faiss-cpu"
        ) from FAISS_IMPORT_ERROR

    namespace = build_cache_namespace(settings, document_text)
    index_path = settings.cache_dir / f"{namespace}.faiss"
    chunks_path = settings.cache_dir / f"{namespace}.chunks.json"

    embedding_provider = build_embedding_provider(settings)

    if not rebuild and index_path.exists() and chunks_path.exists():
        chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=int(item["chunk_id"]),
                text=str(item["text"]),
                start_char=int(item["start_char"]),
                end_char=int(item["end_char"]),
            )
            for item in chunks_payload
        ]
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks):
            raise RuntimeError("FAISS 索引与切片缓存数量不一致，请使用 --rebuild-index")
        bm25 = BM25Index(
            [tokenize_for_retrieval(chunk.text) for chunk in chunks]
        )
        print(f"[索引] 已从缓存加载 {len(chunks)} 个文档切片")
        return HybridRagIndex(
            chunks=chunks,
            index=index,
            bm25=bm25,
            embedding_provider=embedding_provider,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
        )

    chunks = chunk_markdown(
        document_text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    print(
        f"[索引] 正在构建索引：{len(chunks)} 个切片，"
        f"每条最多 {settings.chunk_size} 字符"
    )
    index = HybridRagIndex.build(
        chunks,
        embedding_provider,
        vector_weight=settings.vector_weight,
        bm25_weight=settings.bm25_weight,
    )
    persist_index(
        index,
        namespace=namespace,
        cache_dir=settings.cache_dir,
    )
    print("[索引] FAISS 索引与切片缓存已保存")
    return index


# ---------------------------------------------------------------------------
# 【核心模块｜模型供应商适配】
# 以后的 FastAPI 服务不应直接依赖 OpenAI SDK 或某个供应商的响应结构。
# 所有 LLM 请求统一经过该接口，实现模型替换、超时、重试和审计。
# ---------------------------------------------------------------------------
class LLMProvider(Protocol):
    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        """接收标准 chat messages，返回纯文本回答。"""


class OpenAICompatibleLLMProvider:
    """
    【核心模块｜生产模型适配器】

    使用标准 Chat Completions 协议。只要供应商兼容该协议，
    更换模型时只需修改环境变量，不需要修改 RAG 或检索代码。
    """

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

    def _request_sync(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        response = _post_json(
            url=self.api_url,
            payload=payload,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM API 未返回有效 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            # 某些兼容供应商会返回分段内容数组，这里提取 text 字段。
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM API 返回了空回答")
        return content.strip()

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        # urllib 是同步 API，放入线程池，避免阻塞 asyncio 事件循环。
        return await asyncio.to_thread(
            self._request_sync,
            messages,
            temperature=temperature,
        )


class MockLLMProvider:
    """
    【离线测试模块】

    不做真实生成，只把检索上下文压缩成可读摘要。
    它的价值是无需 API Key 即可验证：
    混合检索是否命中、提示词是否拼装、CLI 是否能完整执行。
    """

    async def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        user_content = messages[-1]["content"] if messages else ""
        match = re.search(r"【用户问题】\s*(.+)$", user_content, flags=re.S)
        question = match.group(1).strip() if match else "未识别问题"
        snippets = re.findall(r"\[资料(\d+)\]\s*(.+?)(?=\n\n\[资料|\Z)", user_content, flags=re.S)
        lines = [
            "【离线 Mock 回答】",
            f"问题：{question}",
            "以下回答仅用于验证检索与提示词链路，不是正式 LLM 生成结果：",
        ]
        for number, snippet in snippets[:3]:
            clean = re.sub(r"\s+", " ", snippet).strip()
            lines.append(f"【资料{number}】{clean[:180]}")
        if not snippets:
            lines.append("未检索到足够资料，请补充制度文档或转人工。")
        return "\n".join(lines)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """
    【核心工厂函数｜供应商可替换】

    增加新供应商时，在这里实现新的 Provider；
    上层 call_llm() 和问答流程不需要修改。
    """

    if settings.llm_provider == "mock":
        print("[提示] 当前使用 Mock LLM。配置 LLM_PROVIDER=openai_compatible 可调用真实模型。")
        return MockLLMProvider()

    if settings.llm_provider == "openai_compatible":
        if not settings.llm_api_url:
            raise ValueError("LLM_PROVIDER=openai_compatible 时必须配置 LLM_API_URL")
        if not settings.llm_api_key:
            raise ValueError("LLM_PROVIDER=openai_compatible 时必须配置 LLM_API_KEY")
        if not settings.llm_model:
            raise ValueError("LLM_PROVIDER=openai_compatible 时必须配置 LLM_MODEL")
        return OpenAICompatibleLLMProvider(
            api_url=settings.llm_api_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.request_timeout,
        )

    raise ValueError(f"不支持的 LLM_PROVIDER: {settings.llm_provider}")


async def call_llm(
    messages: Sequence[dict[str, str]],
    *,
    settings: Settings,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> str:
    """
    【核心模块｜后续 FastAPI 直接复用】

    LLM 调用被封装为独立函数，负责：
    - 通过 Provider 工厂选择供应商；
    - 失败时执行有限次指数退避重试；
    - 对最终异常增加可读上下文；
    - 不把密钥或完整提示词写入控制台。
    """

    provider = create_llm_provider(settings)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await provider.generate(messages, temperature=temperature)
        except Exception as exc:  # API 错误在原型层统一转换为可读异常
            last_error = exc
            if attempt >= max_retries:
                break
            delay = 1.5 * (2**attempt)
            print(
                f"[LLM] 第 {attempt + 1} 次调用失败，"
                f"{delay:.1f} 秒后重试：{exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"LLM 调用失败：{last_error}") from last_error


def _post_json(
    *,
    url: str,
    payload: dict,
    api_key: str,
    timeout: float,
) -> dict:
    """
    【基础设施工具】

    使用标准库发送 JSON 请求，避免为了让单文件原型可运行而安装额外 SDK。
    后续服务端建议替换为异步 httpx.AsyncClient，并接入统一 Trace ID。
    """

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"模型接口返回 HTTP {exc.code}: {detail[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接模型接口: {exc.reason}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("模型接口返回了非 JSON 内容") from exc
    if not isinstance(result, dict):
        raise RuntimeError("模型接口返回结构不是 JSON 对象")
    return result


def build_prompt(question: str, hits: Sequence[SearchHit]) -> list[dict[str, str]]:
    """
    【核心模块｜RAG Prompt 构建】

    系统提示词明确要求：
    - 只能依据当前检索片段回答；
    - 每个主要结论标注【资料N】；
    - 信息不足时明确说明，不得编造；
    - 涉及薪酬承诺、敏感个案的结论提示人工确认。
    """

    context_blocks = []
    for number, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[资料{number}] "
            f"(切片ID={hit.chunk.chunk_id}, "
            f"向量分={hit.vector_score:.4f}, "
            f"BM25分={hit.bm25_score:.4f})\n"
            f"{hit.chunk.text}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "无可用资料。"

    system_prompt = (
        "你是企业员工薪酬福利咨询助手。"
        "只能依据用户提供的知识片段回答，不得补充未出现的政策事实。"
        "回答应简洁、可执行，并用【资料N】标注依据。"
        "如果资料不足、存在冲突或涉及个人薪酬承诺，"
        "必须明确说明并建议创建工单转人工确认。"
    )
    user_prompt = (
        "【知识片段】\n"
        f"{context}\n\n"
        "【用户问题】\n"
        f"{question}\n\n"
        "请用中文回答，并在关键结论后标注来源编号。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def answer_question(
    question: str,
    *,
    rag_index: HybridRagIndex,
    settings: Settings,
) -> tuple[str, list[SearchHit]]:
    """
    【核心模块｜后续 FastAPI 问答接口直接复用】

    把“检索 -> 构建提示词 -> 调用模型”串成一条可测试业务函数。
    Web 层后续只负责鉴权、参数校验和 SSE 输出。
    """

    hits = rag_index.search(question, top_k=settings.top_k)
    if not hits:
        return "未检索到相关资料，请补充知识文档或转人工处理。", []

    messages = build_prompt(question, hits)
    answer = await call_llm(messages, settings=settings)
    return answer, hits


def print_hits(hits: Sequence[SearchHit]) -> None:
    """在命令行打印召回依据，便于第一周验证混合检索是否有效。"""

    print("\n[检索结果]")
    for number, hit in enumerate(hits, start=1):
        preview = re.sub(r"\s+", " ", hit.chunk.text).strip()[:120]
        print(
            f"{number}. 切片 #{hit.chunk.chunk_id} | "
            f"综合={hit.hybrid_score:.4f} | "
            f"向量={hit.vector_score:.4f} | "
            f"BM25={hit.bm25_score:.4f}\n"
            f"   {preview}..."
        )
    print()


async def async_main(args: argparse.Namespace) -> None:
    """
    命令行入口。

    --question 适合自动化测试；不传时进入连续问答模式，输入 exit/quit 退出。
    """

    settings = Settings.from_env(
        document_path=Path(args.document).expanduser().resolve(),
        top_k=args.top_k,
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
    )
    document_text = load_markdown(settings.document_path)
    rag_index = load_or_build_index(
        document_text,
        settings,
        rebuild=args.rebuild_index,
    )

    print(f"[配置] 文档：{settings.document_path}")
    print(f"[配置] 切片：{settings.chunk_size} 字符，重叠 {settings.chunk_overlap}")
    print(
        f"[配置] 混合检索：向量 {settings.vector_weight:.0%} + "
        f"BM25 {settings.bm25_weight:.0%}，Top-{settings.top_k}"
    )
    print(
        f"[配置] LLM：{settings.llm_provider} / "
        f"{settings.llm_model or '内置 Mock'}"
    )
    print(
        f"[配置] Embedding：{settings.embedding_provider} / "
        f"{settings.embedding_model or 'hash'}"
    )

    if args.question:
        questions = [args.question]
    else:
        print("\n输入问题开始咨询。输入 exit 或 quit 退出。")
        questions = None

    while True:
        if questions is not None:
            if not questions:
                break
            question = questions.pop(0)
        else:
            try:
                question = input("\n问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            if question.lower() in {"exit", "quit"}:
                print("已退出。")
                break
            if not question:
                continue

        answer, hits = await answer_question(
            question,
            rag_index=rag_index,
            settings=settings,
        )
        print_hits(hits)
        print("[回答]")
        print(answer)


def parse_args() -> argparse.Namespace:
    """解析最小原型需要的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="员工薪酬福利咨询 - 本地 RAG 命令行原型"
    )
    parser.add_argument(
        "--document",
        default=str(DEFAULT_DOCUMENT),
        help="本地 Markdown 制度文件路径",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="单次提问；不传则进入交互模式",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="混合检索返回条数，默认 3",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="FAISS 和切片缓存目录",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="忽略缓存并重建 FAISS 索引",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 整个程序使用 asyncio 入口，与后续 FastAPI 的异步调用方式保持一致。
    asyncio.run(async_main(parse_args()))
