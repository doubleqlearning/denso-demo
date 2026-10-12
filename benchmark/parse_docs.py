"""Step 1 — parse the source manuals into section-aware chunks with provenance.

    uv run python benchmark/parse_docs.py

PDF → text-layer check → text/tables with layout → section path from bookmarks → header/footer
cleanup → 300–700 token chunks within a section → processed/{documents,chunks}.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

from common import CHUNKS_PATH, DOCUMENTS_PATH, SOURCES, write_jsonl

DOC_IDS = {
    "i658_i4l_series_users_manual_en.pdf": "omron_i4l",
    "sh030166engf.pdf": "melservo_je",
    "sh082625engh.pdf": "fx5_comm",
}

MIN_TOKENS, MAX_TOKENS = 300, 700
MARGIN_TOP, MARGIN_BOTTOM, SIDE_TAB_X = 0.08, 0.92, 0.93

TOPIC_KEYWORDS = {
    "alarm": ["alarm"],
    "error": ["error", "fault", "abnormal"],
    "troubleshooting": ["troubleshoot", "trouble shooting", "corrective action", "remedy", "cause"],
    "maintenance": ["maintenance", "inspection", "replace", "replacement", "lubricat", "service life"],
    "warning": ["warning", "caution"],
    "motor": ["motor", "overload", "overheat", "brake", "regenerat"],
    "encoder": ["encoder"],
    "motion": ["motion", "acceleration", "deceleration", "speed", "position", "torque", "vibration"],
    "safety": ["safety", "emergency stop", "e-stop", "hazard"],
    "communication": ["communication", "ethernet", "modbus", "serial", "time-out", "timeout", "network"],
}
# Sections whose chunks never seed a scenario (front/back matter, register maps).
BOILERPLATE = re.compile(
    r"^(contents|index|revisions?|warranty|trademarks|information and services|relevant manuals|terms|"
    r"generic terms|about the manual|introduction|contact|safety precautions|conditions of use)",
    re.I,
)


def n_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Item:
    page: int  # 1-based
    y: float
    text: str
    is_table: bool = False


# ---------------------------------------------------------------- extraction


def table_markdown(rows: list[list[str | None]]) -> list[str]:
    """One Markdown row per table row; merged (None) and repeated cells collapse."""
    out = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            if cell is None:
                continue
            cell = re.sub(r"\s+", " ", cell).strip()
            if cell and (not cells or cells[-1] != cell):
                cells.append(cell)
        if cells:
            out.append("| " + " | ".join(cells) + " |")
    return out


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        elif text:
            text += ("\n" if line.startswith(("•", "-", "*")) or re.match(r"^\(?\d+[.)]\s", line) else " ") + line
        else:
            text = line
    return text


def page_lines(page: pymupdf.Page) -> list[tuple[float, float, float, str, int]]:
    """(y0, x0, size, text, block_no) for horizontal lines outside side tabs."""
    out = []
    width = page.rect.width
    for b_no, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            if abs(line["dir"][0] - 1) > 0.01 or line["bbox"][0] > SIDE_TAB_X * width:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                size = max(span["size"] for span in line["spans"])
                out.append((line["bbox"][1], line["bbox"][0], size, text, b_no, line["bbox"]))
    return out


def margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.lower())).strip()


def is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"[\dA-Z]{0,2}\s*[-–.]?\s*\d{1,4}|[A-Z]{1,4}", text.strip()))


def extract_document(doc: pymupdf.Document) -> tuple[list[Item], list[int]]:
    raw: list[tuple[int, pymupdf.Page, list]] = []
    margin_counts: Counter[str] = Counter()
    needs_ocr: list[int] = []
    for p_no, page in enumerate(doc, start=1):
        lines = page_lines(page)
        if sum(len(l[3]) for l in lines) < 20:
            if page.get_images():
                needs_ocr.append(p_no)
            continue
        h = page.rect.height
        for line in lines:
            if line[0] < MARGIN_TOP * h or line[0] > MARGIN_BOTTOM * h:
                margin_counts[margin_key(line[3])] += 1
        raw.append((p_no, page, lines))

    items: list[Item] = []
    for p_no, page, lines in raw:
        h = page.rect.height
        try:
            tables = [t for t in page.find_tables().tables if t.row_count >= 2]
        except Exception:
            tables = []
        boxes = [pymupdf.Rect(t.bbox) for t in tables]
        for table, box in zip(tables, boxes):
            rows = table_markdown(table.extract())
            if rows:
                items.append(Item(p_no, box.y0, "\n".join(rows), is_table=True))
        blocks: dict[int, list] = {}
        for y0, x0, size, text, b_no, bbox in lines:
            in_margin = y0 < MARGIN_TOP * h or y0 > MARGIN_BOTTOM * h
            if in_margin and (margin_counts[margin_key(text)] >= 3 or is_page_number(text)):
                continue
            if any(box.intersects(pymupdf.Rect(bbox)) and box.contains(pymupdf.Rect(bbox).tl) for box in boxes):
                continue
            blocks.setdefault(b_no, []).append((y0, text))
        page_items: list[Item] = []
        for block in sorted(blocks.values(), key=lambda b: b[0][0]):
            text = join_lines([t for _, t in block])
            prev = page_items[-1] if page_items else None
            # Some manuals emit one block per visual line: re-join sentence continuations.
            if prev and prev.text.endswith("-") and text[:1].islower():
                prev.text = prev.text + text
            elif prev and not re.search(r"[.:;!?)\]]$", prev.text) and text[:1].islower():
                prev.text = prev.text + " " + text
            else:
                page_items.append(Item(p_no, block[0][0], text))
        items.extend(it for it in page_items if re.search(r"[A-Za-z]{2}", it.text))
    items.sort(key=lambda it: (it.page, round(it.y)))
    return items, needs_ocr


# ---------------------------------------------------------------- sections


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.replace("\xa0", " ").lower()).strip()


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("\xa0", " ")).strip()


def assign_sections(items: list[Item], toc: list[list]) -> list[int]:
    """Index of the TOC entry each item belongs to (-1 before the first bookmark)."""
    by_page: dict[int, list[int]] = {}
    for idx, (_, title, page) in enumerate(toc):
        by_page.setdefault(page, []).append(idx)
    starts: dict[int, list[tuple[float, int]]] = {}
    for page, entries in by_page.items():
        page_items = [it for it in items if it.page == page]
        found = []
        for idx in entries:
            title = norm_title(toc[idx][1])
            # Heading line: an item whose normalized text is part of the bookmark title.
            y = next((it.y for it in page_items
                      if len(norm_title(it.text)) >= 3 and norm_title(it.text) in title
                      and len(it.text) < len(toc[idx][1]) + 25), None)
            found.append((y if y is not None else -1.0, idx))
        # Keep bookmark order: an entry never starts above the previous one.
        last = -1.0
        fixed = []
        for y, idx in found:
            last = max(last, y)
            fixed.append((last, idx))
        starts[page] = fixed

    out, current = [], -1
    for it in items:
        for y, idx in starts.get(it.page, []):
            if y <= it.y + 1 and idx > current:
                current = idx
        out.append(current)
        # Entries on this page below the item stay pending until reached.
    return out


def section_path(toc: list[list], idx: int) -> list[str]:
    if idx < 0:
        return ["Front matter"]
    path, level = [], toc[idx][0] + 1
    for j in range(idx, -1, -1):
        if toc[j][0] < level:
            path.append(clean_title(toc[j][1]))
            level = toc[j][0]
    path.reverse()
    return path


# ---------------------------------------------------------------- chunking


def split_long(item: Item) -> list[Item]:
    """Pack lines (long lines split into sentences) into pieces of at most MAX_TOKENS."""
    if n_tokens(item.text) <= MAX_TOKENS:
        return [item]
    units: list[str] = []
    for line in item.text.split("\n"):
        if n_tokens(line) <= MAX_TOKENS // 2:
            units.append(line)
            continue
        sentences = re.split(r"(?<=[.!?;])\s+", line)
        for sentence in sentences:
            while n_tokens(sentence) > MAX_TOKENS // 2:  # no sentence boundary: hard cut on a space
                cut = sentence.rfind(" ", 0, MAX_TOKENS * 2)
                cut = cut if cut > 0 else MAX_TOKENS * 2
                units.append(sentence[:cut])
                sentence = sentence[cut:].strip()
            units.append(sentence)
    header = units[0] if item.is_table else None
    parts, buf = [], []
    for unit in units:
        if buf and n_tokens("\n".join(buf + [unit])) > MAX_TOKENS - 20:
            parts.append(Item(item.page, item.y, "\n".join(buf), item.is_table))
            buf = [header] if header and n_tokens(header) < 80 else []
        buf.append(unit)
    if buf:
        parts.append(Item(item.page, item.y, "\n".join(buf), item.is_table))
    return parts


def chunk_document(doc_id: str, items: list[Item], toc: list[list]) -> list[dict]:
    sections = assign_sections(items, toc)
    groups: list[tuple[int, list[Item]]] = []
    for it, sec in zip(items, sections):
        if re.search(r"\.{6,}", it.text):
            continue  # table-of-contents lines with dot leaders
        if not groups or groups[-1][0] != sec:
            groups.append((sec, []))
        title = norm_title(section_path(toc, sec)[-1])
        if len(groups[-1][1]) < 2 and norm_title(it.text) and norm_title(it.text) in title:
            continue  # the heading line itself; re-added as "## title"
        groups[-1][1].extend(split_long(it))

    def top(sec: int) -> tuple[str, ...]:
        return tuple(section_path(toc, sec)[:1])

    chunks: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current and current["parts"]:
            chunks.append(current)
        current = None

    for sec, group in groups:
        path = section_path(toc, sec)
        heading = "## " + path[-1]
        if not group:
            continue
        for i, it in enumerate(group):
            piece = (heading + "\n" if i == 0 else "") + it.text
            if current and (top(current["sec"]) != top(sec)
                            or n_tokens(current["text"] + "\n\n" + piece) > MAX_TOKENS
                            or (current["sec"] != sec and n_tokens(current["text"]) >= MIN_TOKENS)):
                flush()
            if current is None:
                current = {"sec": sec, "path": path, "parts": [], "text": "", "pages": [], "table": False}
            current["parts"].append(piece)
            current["text"] = "\n\n".join(current["parts"])
            current["pages"].append(it.page)
            current["table"] |= it.is_table
    flush()

    # Merge a too-small chunk into its predecessor when both share the top-level section.
    merged: list[dict] = []
    for c in chunks:
        if (merged and n_tokens(c["text"]) < MIN_TOKENS // 2 and top(merged[-1]["sec"]) == top(c["sec"])
                and n_tokens(merged[-1]["text"] + c["text"]) <= MAX_TOKENS + 50):
            prev = merged[-1]
            prev["text"] += "\n\n" + c["text"]
            prev["pages"] += c["pages"]
            prev["table"] |= c["table"]
        else:
            merged.append(c)

    out, per_page = [], Counter()
    for c in merged:
        page_start = min(c["pages"])
        per_page[page_start] += 1
        text = c["text"].strip()
        topics = match_topics(" ".join(c["path"]) + " " + text)
        out.append({
            "chunk_id": f"{doc_id}_p{page_start:04d}_c{per_page[page_start]:02d}",
            "doc_id": doc_id,
            "section_path": c["path"],
            "page_start": page_start,
            "page_end": max(c["pages"]),
            "text": text,
            "n_tokens": n_tokens(text),
            "has_table": c["table"],
            "topics": topics,
            "scenario_candidate": is_candidate(c["path"], text, topics),
        })
    return out


def match_topics(text: str) -> list[str]:
    low = text.lower()
    return [topic for topic, words in TOPIC_KEYWORDS.items() if any(w in low for w in words)]


def is_candidate(path: list[str], text: str, topics: list[str]) -> bool:
    if any(BOILERPLATE.match(re.sub(r"^(section|part|chapter)?\s*[\d\-.]*\s*", "", p, flags=re.I)) for p in path):
        return False
    if n_tokens(text) < 120 or len(re.findall(r"[a-z]{3,}", text)) < 60:
        return False  # mostly numbers / register maps
    path_topics = match_topics(" ".join(path))
    # Either the section is about a target topic, or the text is about at least two of them.
    return bool(path_topics) or len(topics) >= 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    documents, chunks = [], []
    for pdf in sorted(SOURCES.glob("*.pdf")):
        doc_id = DOC_IDS.get(pdf.name, re.sub(r"[^a-z0-9]+", "_", pdf.stem.lower()))
        doc = pymupdf.open(pdf)
        toc = doc.get_toc()
        items, needs_ocr = extract_document(doc)
        doc_chunks = chunk_document(doc_id, items, toc)
        chunks.extend(doc_chunks)
        documents.append({
            "doc_id": doc_id,
            "file": pdf.name,
            "title": doc.metadata.get("title") or pdf.stem,
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "n_pages": doc.page_count,
            "needs_ocr_pages": needs_ocr,
            "n_chunks": len(doc_chunks),
            "n_candidate_chunks": sum(c["scenario_candidate"] for c in doc_chunks),
            "toc": [{"level": lvl, "title": clean_title(t), "page": p} for lvl, t, p in toc],
        })
        sizes = [c["n_tokens"] for c in doc_chunks]
        print(f"{doc_id}: {doc.page_count} pages, {len(doc_chunks)} chunks "
              f"(tokens min {min(sizes)} / median {sorted(sizes)[len(sizes)//2]} / max {max(sizes)}), "
              f"{documents[-1]['n_candidate_chunks']} scenario candidates, {len(needs_ocr)} pages need OCR")
    write_jsonl(DOCUMENTS_PATH, documents)
    write_jsonl(CHUNKS_PATH, chunks)
    print(f"wrote {DOCUMENTS_PATH} and {CHUNKS_PATH} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
