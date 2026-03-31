"""
File readers for all supported document types.
Each reader returns a list of Section objects — one per natural unit
(page, slide, chapter, heading-section, etc.).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Section:
    text: str
    phys_page: int  # 1-based index within the document
    page_label: str  # human-readable label used in citations
    total: int  # total sections in this document


def _make_sections(parts: list[tuple[str, str]]) -> list[Section]:
    """Build Section list from (text, label) pairs."""
    total = len(parts)
    return [
        Section(text=t, phys_page=i + 1, page_label=label, total=total)
        for i, (t, label) in enumerate(parts)
    ]


# ── PDF ───────────────────────────────────────────────────────────────────────


def read_pdf(path: Path) -> list[Section]:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    total = len(doc)
    sections = []
    for page_num in range(total):
        page = doc[page_num]
        text = page.get_text().strip()
        phys_page = page_num + 1
        label = page.get_label() or str(phys_page)
        sections.append(Section(text=text, phys_page=phys_page, page_label=label, total=total))
    doc.close()
    return sections


# ── Plain text ────────────────────────────────────────────────────────────────


def read_text(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Section(text=text, phys_page=1, page_label="1", total=1)]


# ── Markdown — split by ATX headings (#, ##, …) ──────────────────────────────


def read_markdown(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8", errors="replace")
    heading_re = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    parts = _split_by_headings(text, heading_re)
    return _make_sections(parts) if parts else read_text(path)


# ── reStructuredText — split by underlined section titles ────────────────────


def read_rst(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    parts: list[tuple[str, str]] = []
    current: list[str] = []
    current_label = "intro"
    underline_re = re.compile(r"^[=\-~^\"'`#*+]{2,}$")
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        is_underline = next_line and underline_re.match(next_line) and len(next_line) >= len(line)
        if line.strip() and is_underline:
            if current:
                parts.append(("\n".join(current).strip(), current_label))
            current_label = line.strip()[:60]
            current = [line, next_line]
            i += 2
        else:
            current.append(line)
            i += 1
    if current:
        parts.append(("\n".join(current).strip(), current_label))
    parts = [(t, lbl) for t, lbl in parts if t.strip()]
    return _make_sections(parts) if parts else read_text(path)


def _split_by_headings(text: str, heading_re: re.Pattern) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    current: list[str] = []
    current_label = "intro"
    for line in text.splitlines():
        m = heading_re.match(line)
        if m:
            if current:
                parts.append(("\n".join(current).strip(), current_label))
            current_label = m.group(1).strip()[:60]
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append(("\n".join(current).strip(), current_label))
    return [(t, lbl) for t, lbl in parts if t.strip()]


# ── HTML ──────────────────────────────────────────────────────────────────────


def read_html(path: Path) -> list[Section]:
    from bs4 import BeautifulSoup, NavigableString, Tag

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    heading_tags = {"h1", "h2", "h3", "h4"}
    default_label = (
        soup.title.string.strip()[:60] if soup.title and soup.title.string else path.stem
    )

    parts: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_label = default_label

    root = soup.body or soup
    for node in root.descendants:
        if isinstance(node, Tag) and node.name in heading_tags:
            text = "\n".join(current_lines).strip()
            if text:
                parts.append((text, current_label))
            current_label = node.get_text(" ", strip=True)[:60] or current_label
            current_lines = []
        elif isinstance(node, NavigableString) and not any(
            isinstance(p, Tag) and p.name in heading_tags for p in node.parents
        ):
            line = str(node).strip()
            if line:
                current_lines.append(line)

    text = "\n".join(current_lines).strip()
    if text:
        parts.append((text, current_label))

    parts = [(t, lbl) for t, lbl in parts if t.strip()]
    return _make_sections(parts) if parts else read_text(path)


# ── DOCX ──────────────────────────────────────────────────────────────────────


def read_docx(path: Path) -> list[Section]:
    from docx import Document

    doc = Document(str(path))
    parts: list[tuple[str, str]] = []
    current: list[str] = []
    current_label = "intro"

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading") and para.text.strip():
            if current:
                parts.append(("\n".join(current).strip(), current_label))
            current_label = para.text.strip()[:60]
            current = []
        elif para.text.strip():
            current.append(para.text.strip())

    if current:
        parts.append(("\n".join(current).strip(), current_label))

    parts = [(t, lbl) for t, lbl in parts if t.strip()]
    return _make_sections(parts) if parts else read_text(path)


# ── PPTX ──────────────────────────────────────────────────────────────────────


def read_pptx(path: Path) -> list[Section]:
    from pptx import Presentation

    prs = Presentation(str(path))
    total = len(prs.slides)
    sections = []

    for i, slide in enumerate(prs.slides):
        lines: list[str] = []
        title = ""
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                line = para.text.strip()
                if line:
                    lines.append(line)
            if not title and shape.text.strip():
                title = shape.text.strip()[:60]

        text = "\n".join(lines).strip()
        label = title or f"Slide {i + 1}"
        sections.append(Section(text=text, phys_page=i + 1, page_label=label, total=total))

    return sections


# ── EPUB ──────────────────────────────────────────────────────────────────────


def read_epub(path: Path) -> list[Section]:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    parts: list[tuple[str, str]] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        if not text:
            continue
        heading = soup.find(["h1", "h2", "h3"])
        label = heading.get_text().strip()[:60] if heading else item.get_name()
        parts.append((text, label))

    return _make_sections(parts) if parts else read_text(path)


# ── Registry ──────────────────────────────────────────────────────────────────

READERS: dict[str, Callable[[Path], list[Section]]] = {
    ".pdf": read_pdf,
    ".txt": read_text,
    ".md": read_markdown,
    ".rst": read_rst,
    ".html": read_html,
    ".htm": read_html,
    ".docx": read_docx,
    ".pptx": read_pptx,
    ".epub": read_epub,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(READERS)


def read_file(path: Path) -> list[Section]:
    ext = path.suffix.lower()
    if ext not in READERS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type: {ext}. Supported: {supported}")
    return READERS[ext](path)
