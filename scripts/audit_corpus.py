"""Audit locally downloaded PDFs against the candidate corpus manifest."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "evaluations" / "corpus_audit.json"

def audit_one(meta: dict) -> dict:
    path = RAW / meta["file_name"]
    result = {"id": meta["id"], "filename": meta["file_name"], "exists": path.is_file()}
    if not result["exists"]:
        return result
    data = path.read_bytes()
    result["sha256"] = hashlib.sha256(data).hexdigest()
    result["size_bytes"] = len(data)
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            chars = [len(page.get_text("text").strip()) for page in doc]
            result["page_count"] = len(doc)
            result["total_extracted_chars"] = sum(chars)
            result["pages_over_100_chars"] = sum(v >= 100 for v in chars)
            result["pages_low_text"] = [i + 1 for i, v in enumerate(chars) if v < 100]
            result["text_page_fraction"] = round(result["pages_over_100_chars"] / len(doc), 3) if doc else 0
            result["requires_manual_inspection"] = bool(result["pages_low_text"])
    except Exception as exc:
        result["error"] = str(exc)
    return result

def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [audit_one(meta) for meta in data["documents"]]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "PyMuPDF get_text('text'); pages <100 chars marked for inspection (heuristic, not a pass/fail grade)",
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {OUT}")
    for r in records:
        if not r["exists"]:
            print(f"{r['id']}: MISSING — download {r['filename']}")
        elif r.get("error"):
            print(f"{r['id']}: INVALID PDF — {r['error']}")
        else:
            print(f"{r['id']}: {r['page_count']} pages, "
                  f"{r['total_extracted_chars']} chars, "
                  f"{len(r['pages_low_text'])} low-text pages, SHA256={r['sha256'][:12]}...")
    return 0 if all(r["exists"] and not r.get("error") for r in records) else 1

if __name__ == "__main__":
    raise SystemExit(main())