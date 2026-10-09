import io
from pathlib import PurePath

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.errors import AppError

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}


def extract_text(filename: str, data: bytes) -> str:
    """Turn an uploaded file into plain text."""
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AppError(
            415,
            "unsupported_file_type",
            f"Unsupported file type '{ext or 'none'}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise AppError(422, "invalid_pdf", f"Could not read PDF: {exc}") from exc
    else:
        text = data.decode("utf-8", errors="replace")

    text = text.strip()
    if not text:
        raise AppError(422, "empty_document", "No extractable text found in the file.")
    return text


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 2),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]
