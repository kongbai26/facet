"""Prepare a small, reproducible HotpotQA subset for Facet evaluation.

The official dataset is downloaded separately because it is large and has its
own license.  This converter keeps the original question/supporting-fact
provenance and emits plain-text documents plus the repository's versioned
evaluation-case format.  The generated ``doc_id`` values are deterministic so
the same subset can be ingested and evaluated repeatedly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:80] or "untitled"


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("HotpotQA input must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def prepare(records: list[dict[str, Any]], output_dir: Path, limit: int) -> dict[str, Any]:
    selected = records[:limit] if limit > 0 else records
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    documents: list[dict[str, str]] = []

    for record in selected:
        record_id = str(record.get("_id") or record.get("id") or "").strip()
        question = str(record.get("question") or "").strip()
        answer = str(record.get("answer") or "").strip()
        contexts = record.get("context") or []
        supporting = record.get("supporting_facts") or []
        if not record_id or not question or not contexts:
            continue

        title_to_doc_id: dict[str, str] = {}
        for context in contexts:
            if not isinstance(context, list) or len(context) != 2:
                continue
            title, sentences = str(context[0]), context[1]
            if not isinstance(sentences, list):
                continue
            doc_id = f"hotpot-{_slug(title)}"
            title_to_doc_id[title] = doc_id
            filename = f"{doc_id}.txt"
            text = f"# {title}\n\n" + "\n".join(str(sentence).strip() for sentence in sentences)
            path = documents_dir / filename
            if not path.exists():
                path.write_text(text + "\n", encoding="utf-8")
            documents.append({"doc_id": doc_id, "filename": filename, "title": title})

        expected_doc_ids = [title_to_doc_id[str(item[0])] for item in supporting if isinstance(item, list) and item and str(item[0]) in title_to_doc_id]
        if not expected_doc_ids:
            continue
        cases.append(
            {
                "id": f"hotpot_{record_id}",
                "query": question,
                "category": str(record.get("type") or "multi_hop"),
                "difficulty": str(record.get("level") or "standard"),
                "expected_doc_ids": list(dict.fromkeys(expected_doc_ids)),
                "expected_facts": [[answer]] if answer else [],
                "must_cite": True,
                "provenance": {"source_id": record_id, "supporting_facts": supporting},
            }
        )

    unique_documents = {item["doc_id"]: item for item in documents}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "HotpotQA",
                "license": "CC BY-SA 4.0; verify source terms before redistribution",
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(
            {"source": "HotpotQA", "documents": list(unique_documents.values()), "case_count": len(cases)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"cases": len(cases), "documents": len(unique_documents), "output": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert official HotpotQA JSON to Facet evaluation files.")
    parser.add_argument("input", type=Path, help="Official HotpotQA JSON file")
    parser.add_argument("output_dir", type=Path, help="Directory for documents and evaluation.json")
    parser.add_argument("--limit", type=int, default=20, help="Number of questions to convert; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    print(json.dumps(prepare(_load_records(args.input), args.output_dir, args.limit), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
