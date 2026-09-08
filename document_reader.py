"""
document_reader.py
------------------
Step 1 of the pipeline: turn a file on disk into plain text.

We support three formats and use a different library for each:

    .txt   -> just read the file
    .pdf   -> pypdf
    .docx  -> python-docx

Everything else in the project only ever sees plain text, so adding a new
format later means adding one small function here and nothing else.
"""

from pathlib import Path

import config


def find_documents(folder: Path) -> list[Path]:
    """Return every supported document in `folder`, sorted by file name.

    Files we cannot read (a .csv, a .xlsx, a hidden file) are simply skipped
    instead of crashing the program.
    """
    if not folder.exists():
        raise FileNotFoundError(f"The input folder does not exist: {folder}")

    documents = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and path.suffix.lower() in config.SUPPORTED_FILE_TYPES
        and not path.name.startswith(".")
    ]
    return documents


def read_document(path: Path) -> str:
    """Read one file and return its text.

    Raises:
        ValueError: if the file type is not supported, or the file has no
            readable text in it (for example a scanned image-only PDF).
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = _read_txt(path)
    elif suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        raise ValueError(f"Cannot read this file type: {suffix}")

    text = text.strip()
    if len(text) < 30:
        raise ValueError("The file contains almost no text (is it a scanned image?)")

    # Cut very long documents so the API call stays cheap and fast.
    if len(text) > config.MAX_CHARACTERS:
        text = text[: config.MAX_CHARACTERS] + "\n\n[document truncated]"

    return text


def _read_txt(path: Path) -> str:
    """Read a plain text file."""
    # errors="replace" means a strange character will not crash the program.
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    """Read a PDF by joining the text of every page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    """Read a Word document by joining every non-empty paragraph."""
    import docx

    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
