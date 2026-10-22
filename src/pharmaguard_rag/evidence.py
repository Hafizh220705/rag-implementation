"""Bound generated answers to complete, attributable excerpts from retrieval.

The LLM selects evidence IDs. It cannot supply new factual prose. This checks
text provenance and output limits, not semantic relevance or answer completeness.
"""

import json
import math
import re

from pharmaguard_rag.citations import resolve_citations
from pharmaguard_rag.retrieval_dense import retrieval_query

DOMAIN_TERMS = {
    "KTD": "Kejadian Tidak Diinginkan",
    "ESO": "Efek Samping Obat",
    "KIPI": "Kejadian Ikutan Pasca Imunisasi",
}
MAX_ANSWER_WORDS = 120
MAX_EXCERPTS = 3
MAX_EVIDENCE_CHARS = 10000
MAX_CANDIDATES_PER_DOCUMENT = 3
STOP_WORDS = set("apa siapa bagaimana mengapa yang dan atau untuk dari dalam dengan menurut pada terkait terhadap antara tersebut adalah itu ini suatu oleh sebagai serta harus perlu telah dapat lebih".split())
INCOMPLETE_END = re.compile(r"\b(?:dan|atau|dapat|harus|untuk|dengan|yang|kepada|termasuk|antara|sebagai)\s*[.;:,]*$", re.I)


def normalize(text: str) -> str:
    return " ".join(text.split())


def terms(text: str) -> set[str]:
    tokens = set(re.findall(r"[\w-]+", text.lower())) - STOP_WORDS
    families = (
        ({"penunjukan", "menunjuk", "ditunjuk", "tunjuk"}, "tunjuk"),
        ({"perubahan", "berubah", "mengubah", "ubah"}, "ubah"),
        ({"pelaporan", "melaporkan", "dilaporkan", "laporan"}, "lapor"),
        ({"menyampaikan", "disampaikan", "penyampaian"}, "sampai"),
        ({"pemantauan", "memantau", "dipantau", "pengawasan", "mengawasi", "pantau"}, "pantau"),
    )
    for words, canonical in families:
        if tokens & words:
            tokens = (tokens - words) | {canonical}
    return tokens


def complete_units(text: str) -> list[dict]:
    """Find complete sentences/list items with inherited literal list scope.

    Text offsets refer to whitespace-normalized source text. Headings and a
    dangling final fragment are excluded. A list item's introduction is kept
    verbatim so an obligation does not lose its named actor.
    """
    text = normalize(text)
    # Mask only the dot of an enumeration marker, preserving source offsets.
    masked = re.sub(r"(?<!\w)(?:\d{1,2}|[a-zA-Z])\.(?=\s)",
                    lambda match: match.group()[:-1] + "\u2024", text)
    boundary = re.compile(r"(?<=[.!?])\s+|(?<=;)\s+(?=[A-Z0-9])|\s+(?=(?:\d{1,2}[\u2024)]|[a-zA-Z][\u2024)])\s+)")
    units = []
    scope = ""
    scope_span = None
    bounds = [0] + [match.end() for match in boundary.finditer(masked)] + [len(text)]
    for unit_index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        part = text[start:end].rstrip()
        cursor = start + len(part)
        page_prefix = re.match(r"(?:-\d+-|\d+|[ivxlcdm]+)\s+", part)
        if page_prefix:
            start += page_prefix.end()
            part = part[page_prefix.end():]
        header = re.match(r"(?:[A-Z][A-Z0-9-]*\s+){2,}(?=[A-Z][a-z])", part)
        if header:
            start += header.end()
            part = part[header.end():]
        part = part.strip()
        if not part:
            continue
        # A list introduction may precede its first numbered entry.
        if part.endswith(":") or re.match(r"^[a-zA-Z]\.\s+[A-Z]", part):
            scope = part
            scope_span = [start, cursor]
            continue
        marker = re.match(r"(?:\d{1,2}[.)]|[a-z][.)])\s+", part)
        if marker:
            if not scope or scope_span is None:
                continue  # Subject/context is unavailable at this boundary.
            item = part[marker.end():].strip()
            next_is_marker = (unit_index + 1 < len(bounds) - 1 and
                              re.match(r"(?:\d{1,2}[.)]|[a-z][.)])\s+", text[bounds[unit_index + 1]:]))
            if (not item.endswith((".", ";", "!", "?")) and not next_is_marker) or INCOMPLETE_END.search(item):
                continue
            # A next list marker establishes a boundary, not a proposition.
            # Unpunctuated titles such as 'Definisi dan Tujuan dari PV' are
            # not evidence. Require an explicit predicate for such items.
            if not item.endswith((".", ";", "!", "?")) and not re.search(
                r"\b(?:adalah|merupakan|harus|wajib|dapat|mem\w+|men\w+|mel\w+|ber\w+)\b",
                item, re.I,
            ):
                continue
            selected_spans = [scope_span, [start + marker.end(), cursor]]
            quote = scope + " … " + item
        else:
            scope = ""
            scope_span = None
            # Drop leading headings/page numbers, but never trim a lowercase
            # chunk continuation into a newly invented sentence.
            if not part[0].isupper() or not part.endswith((".", "!")):
                continue
            if INCOMPLETE_END.search(part) or len(part.split()) < 5:
                continue
            selected_spans = [[start, cursor]]
            quote = part
        # Keep the source spelling (including acronyms). Never expand terms
        # by guessing; canonical expansions are given only in the prompt.
        if len(quote.split()) <= MAX_ANSWER_WORDS - 12:
            units.append({"text": quote, "spans": selected_spans})
    return units


def build_evidence(question: str, pages: list[dict]) -> list[dict]:
    """Pack relevant complete excerpts fairly across retrieved source pages."""
    query_terms = terms(retrieval_query(question))
    # Explicit definitions and requested actions need that actual predicate,
    # not a nearby sentence that merely shares 'farmakovigilans'. These are
    # query-shape rules, never question IDs or prewritten answers.
    definition = re.search(
        r"\b(?:definisi|yang dimaksud dengan|apa itu)\s+(.+?)(?:\s+menurut\b|[?!.]|$)",
        question, re.I,
    )
    action_terms = query_terms & {"tunjuk", "ubah", "sampai"}
    recipient = re.search(r"\bkepada\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
    ongoing_monitoring = (re.search(r"\bmengapa\b", question, re.I)
                          and "pantau" in query_terms
                          and re.search(r"\b(?:terus|berhenti|beredar|setelah)\b", question, re.I))
    all_units = [complete_units(page["text"]) for page in pages]
    unit_terms = [terms(unit["text"]) for units in all_units for unit in units]
    weights = {word: math.log(1 + len(unit_terms) / (1 + sum(word in words for words in unit_terms)))
               for word in query_terms}
    candidates_by_document = {}
    for source_index, page in enumerate(pages, 1):
        candidates = candidates_by_document.setdefault(page["document_id"], [])
        for unit in all_units[source_index - 1]:
            unit_words = terms(unit["text"])
            if definition and not re.search(
                r"(?:^|…\s*)" + re.escape(definition.group(1).strip()) + r"\s+(?:adalah|merupakan)\b",
                unit["text"], re.I,
            ):
                continue
            if action_terms and not action_terms & unit_words:
                continue
            if recipient and recipient.group(1).lower() not in unit["text"].lower():
                continue
            if ongoing_monitoring and not re.search(
                r"\b(?:keterbatasan|belum|tidak semua|tidak terdeteksi|tidak diketahui)\b",
                unit["text"], re.I,
            ):
                # A list of side effects is not a reason for ongoing monitoring.
                # Conservative: abstain if the retrieved pages lack such a reason.
                continue
            overlap = query_terms & unit_words
            score = sum(weights[word] for word in overlap)
            if re.search(r"definisi|dimaksud", question, re.I) and re.search(r"\b(adalah|merupakan)\b", unit["text"], re.I):
                score += 3
            if re.search(r"mengapa", question, re.I) and re.search(r"\b(karena|sehingga|untuk|guna)\b", unit["text"], re.I):
                score += 2
            if re.search(r"mengapa", question, re.I) and re.search(
                r"\b(?:keterbatasan|menyebabkan|belum|tidak semua|tidak terdeteksi)\b", unit["text"], re.I,
            ):
                score += 4
            if not overlap:
                continue
            candidates.append({**unit, "source_label": f"S{source_index}",
                               "document_id": page["document_id"],
                               "source_index": source_index - 1,
                               "relevance_score": score})
    for candidates in candidates_by_document.values():
        candidates.sort(key=lambda item: (-item["relevance_score"], len(item["text"])))
    packed = []
    size = 0
    # Round robin prevents a verbose regulation from consuming the whole budget.
    for rank in range(MAX_CANDIDATES_PER_DOCUMENT):
        for candidates in candidates_by_document.values():
            if rank >= len(candidates):
                continue
            candidate = candidates[rank]
            if size + len(candidate["text"]) > MAX_EVIDENCE_CHARS:
                continue
            size += len(candidate["text"])
            packed.append({**candidate, "evidence_id": f"E{len(packed) + 1}"})
    return packed


def render_selection(raw: str, evidence: list[dict], pages: list[dict],
                     required_documents: list[str]) -> tuple[str, list[dict], list[dict]]:
    """Validate exact selection, coverage, offsets and length before rendering."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Output must be a JSON object with evidence_ids only.") from error
    if not isinstance(payload, dict) or set(payload) != {"evidence_ids"}:
        raise ValueError("Free-form claims are not allowed; return evidence_ids only.")
    ids = payload["evidence_ids"]
    if required_documents:
        if not isinstance(ids, dict) or set(ids) != set(required_documents):
            raise ValueError("Select one evidence ID per required document.")
        lookup = {item["evidence_id"]: item for item in evidence}
        if any(value is None for value in ids.values()):
            return "", [], []
        for doc, value in ids.items():
            if not isinstance(value, str) or value not in lookup or lookup[value]["document_id"] != doc:
                raise ValueError("Evidence does not belong to the claimed document.")
        ids = list(ids.values())
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError("evidence_ids must be a list of strings.")
    if not ids:
        return "", [], []
    if len(ids) > MAX_EXCERPTS or len(ids) != len(set(ids)):
        raise ValueError("Select at most three distinct evidence IDs.")
    lookup = {item["evidence_id"]: item for item in evidence}
    if any(item not in lookup for item in ids):
        raise ValueError("Unknown evidence ID. Do not invent evidence or source labels.")
    selected = [lookup[item] for item in ids]
    missing = set(required_documents) - {item["document_id"] for item in selected}
    if missing:
        raise ValueError("Missing required source(s): " + ", ".join(sorted(missing)))
    answer_parts = []
    for item in selected:
        page = pages[item["source_index"]]
        source_text = normalize(page["text"])
        reconstructed = " … ".join(source_text[start:end] for start, end in item["spans"])
        if reconstructed != item["text"]:
            raise ValueError("Evidence offsets do not match the source text.")
        title = page.get("title") or page["document_id"]
        answer_parts.append(f'- {title}: “{item["text"]}” [{item["source_label"]}]')
    answer = "\n".join(answer_parts)
    # The word budget includes source titles; citation identifiers are excluded.
    if len(re.sub(r"\[S\d+\]", "", answer).split()) > MAX_ANSWER_WORDS:
        raise ValueError("Selection exceeds 120 words including source titles. Select shorter evidence.")
    resolved, sources = resolve_citations(answer, pages)
    for source in sources:
        page = pages[int(source["label"][1:]) - 1]
        source["context_chunk_ids"] = page.get("context_chunk_ids", [page["chunk_id"]])
    return resolved, sources, selected
