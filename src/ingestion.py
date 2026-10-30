"""
制度文档入库服务。

负责 Markdown/PDF 解析、500 字切片、SQLite 保存和 FAISS 索引重建。
HTTP 上传路由只负责文件校验和保存原始文件，实际解析与向量化在后台任务执行。
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import re
import uuid
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

try:
    from . import stage1_rag_cli as rag
    from .db import SQLiteDatabase
    from .rag_runtime import RagRuntime
except ImportError:
    import stage1_rag_cli as rag
    from db import SQLiteDatabase
    from rag_runtime import RagRuntime


SUPPORTED_SUFFIXES = {".md", ".pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024


def safe_filename(filename: str) -> str:
    """移除路径部分和控制字符，避免目录穿越。"""

    base = Path(filename).name
    base = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", "_", base).strip(" .")
    if not base:
        raise ValueError("文件名无效")
    return base[:180]


def validate_upload(filename: str, content: bytes) -> tuple[str, str]:
    """
    校验扩展名、大小和内容。

    返回规范文件名和 MIME 类型。扫描型 PDF 的 OCR 支持放到后续阶段。
    """

    normalized = safe_filename(filename)
    suffix = Path(normalized).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 Markdown 或 PDF 文件")
    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("上传文件不能超过 20 MB")
    mime_type = (
        "text/markdown"
        if suffix == ".md"
        else "application/pdf"
    )
    return normalized, mime_type


def calculate_hash(content: bytes) -> str:
    """使用 SHA-256 对文件内容去重。"""

    return hashlib.sha256(content).hexdigest()


def extract_document_text(path: Path) -> str:
    """
    将 Markdown 或 PDF 解析为纯文本。

    PDF 使用 pypdf 提取文本层；扫描版 PDF 不包含文本层，因此明确报错。
    """

    suffix = path.suffix.lower()
    if suffix == ".md":
        text = path.read_text(encoding="utf-8-sig")
    elif suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(path.read_bytes()))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError("PDF 文件解析失败，请确认文件未损坏") from exc
        if not text.strip():
            raise ValueError("暂不支持没有文本层的扫描版 PDF")
    else:
        raise ValueError("不支持的制度文件类型")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("制度文件没有可用文本")
    return normalized


async def run_ingestion_job(
    *,
    database: SQLiteDatabase,
    runtime: RagRuntime,
    job_id: str,
    document_id: str,
    storage_path: str,
) -> None:
    """
    执行完整入库任务。

    任一步骤失败后同时更新任务和文档状态，并将其他 ready 文档保留在现有索引中。
    """

    try:
        await database.update_ingestion_job(
            job_id=job_id,
            status="parsing",
            stage="解析制度文件",
            progress=10,
        )
        text = await asyncio.to_thread(extract_document_text, Path(storage_path))

        await database.update_ingestion_job(
            job_id=job_id,
            status="chunking",
            stage="按 500 字切片",
            progress=30,
        )
        chunks = rag.chunk_markdown(
            text,
            chunk_size=runtime.settings.chunk_size,
            overlap=runtime.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("制度文件没有生成有效切片")

        await database.replace_document_chunks(
            document_id=document_id,
            chunks=[
                {
                    "chunk_index": chunk.chunk_id,
                    "content": chunk.text,
                    "char_start": chunk.start_char,
                    "char_end": chunk.end_char,
                }
                for chunk in chunks
            ],
        )
        await database.update_ingestion_job(
            job_id=job_id,
            status="embedding",
            stage="生成向量并重建混合索引",
            progress=70,
        )
        await database.update_document_status(
            document_id=document_id,
            status="ready",
        )

        # 重建过程会读取全部 ready 文档，因此新文档与旧文档一起进入索引。
        await runtime.reload_from_database(database)
        await database.update_ingestion_job(
            job_id=job_id,
            status="ready",
            stage="入库完成",
            progress=100,
        )
    except Exception as exc:
        await database.update_document_status(
            document_id=document_id,
            status="failed",
        )
        await database.update_ingestion_job(
            job_id=job_id,
            status="failed",
            stage="入库失败",
            progress=100,
            error_code="INGESTION_FAILED",
            error_message=str(exc) or "制度文件解析或向量化失败。",
        )


async def create_document_job(
    *,
    database: SQLiteDatabase,
    file_path: Path,
    original_filename: str,
    mime_type: str,
    created_by: str,
    title: str | None = None,
    version: str = "1.0",
) -> tuple[str, str]:
    """
    创建文档和入库任务记录，返回 (document_id, job_id)。

    实际向量化由调用方通过 FastAPI BackgroundTasks 启动。
    """

    content = file_path.read_bytes()
    file_hash = calculate_hash(content)
    document_id = hashlib.sha256(
        f"{file_hash}:{version}".encode("utf-8")
    ).hexdigest()[:20]
    document_title = (title or Path(original_filename).stem).strip()
    await database.create_knowledge_document(
        document_id=document_id,
        file_name=original_filename,
        title=document_title,
        version=version,
        file_hash=file_hash,
        storage_path=str(file_path),
        file_size=len(content),
        mime_type=mime_type,
        created_by=created_by,
        status="processing",
    )
    job_id = await database.create_ingestion_job(
        document_id=document_id,
        created_by=created_by,
    )
    return document_id, job_id


async def seed_default_document(
    *,
    database: SQLiteDatabase,
    runtime: RagRuntime,
    source_path: Path,
    upload_dir: Path,
    created_by: str,
) -> None:
    """
    首次启动且数据库没有 ready 文档时导入示例制度。

    已存在有效文档时不做任何操作，避免覆盖管理员后续维护的知识库。
    """

    if not source_path.exists():
        runtime.error_message = f"默认制度文件不存在：{source_path}"
        return

    ready_rows = await database.get_ready_chunks()
    if ready_rows:
        await runtime.reload_from_database(database)
        return

    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix.lower()
    destination = upload_dir / f"seed-{uuid.uuid4().hex[:10]}{suffix}"
    destination.write_bytes(source_path.read_bytes())
    normalized, mime_type = validate_upload(source_path.name, destination.read_bytes())
    existing = await database.find_document_by_hash(calculate_hash(destination.read_bytes()))
    if existing is None:
        document_id, job_id = await create_document_job(
            database=database,
            file_path=destination,
            original_filename=normalized,
            mime_type=mime_type,
            created_by=created_by,
            title=source_path.stem,
            version="1.0",
        )
        await run_ingestion_job(
            database=database,
            runtime=runtime,
            job_id=job_id,
            document_id=document_id,
            storage_path=str(destination),
        )
    else:
        await runtime.reload_from_database(database)
