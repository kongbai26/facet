"""Create an offline backup of Facet runtime data.

Run during a write-idle window.  SQLite uses its online backup API; uploaded
files, Chroma persistence and BM25 caches are copied into the same archive.
Configuration and secrets are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_config


def _copy_tree(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return True


def create_backup(output_dir: Path) -> Path:
    settings = get_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output_dir / f"facet-backup-{stamp}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="facet-backup-", dir=output_dir) as temp_name:
        staging = Path(temp_name)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "created_at": stamp,
            "write_idle_required": True,
            "included": [],
        }

        metadata_db = Path(settings.storage.metadata_db)
        if metadata_db.exists():
            db_copy = staging / "metadata.db"
            source = sqlite3.connect(str(metadata_db))
            try:
                target = sqlite3.connect(str(db_copy))
                try:
                    source.backup(target)
                finally:
                    target.close()
            finally:
                source.close()
            manifest["included"].append("metadata.db")

        paths = {
            "uploads": Path(settings.storage.upload_dir),
            "chroma": Path(settings.vectorstore.persist_dir),
            "bm25_cache": Path(settings.retrieval.hybrid.bm25_cache_dir),
        }
        for name, source in paths.items():
            if _copy_tree(source, staging / name):
                manifest["included"].append(name)

        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging / "manifest.json", arcname="manifest.json")
            for name in manifest["included"]:
                archive.add(staging / str(name), arcname=str(name))
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up Facet runtime data without secrets.")
    parser.add_argument("--output", type=Path, default=Path("./data/backups"))
    args = parser.parse_args()
    archive = create_backup(args.output)
    print(json.dumps({"backup": str(archive), "write_idle_required": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
