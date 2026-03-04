#!/usr/bin/env python3
# pylint: disable=import-error,too-many-locals
"""
import_openai.py

Import ChatGPT export archive content into Couchbase.

Source:
- manifest.json produced by parse_openai.py (chatgpt_export_unpacker)

Target keyspace:
  bucket = library
  scope = openai
  collection = conversations

Credentials:
- loaded from ./script/.env (or env vars)

Usage:
  python ./script/import_openai.py --manifest ./output/manifest.json
  python ./script/import_openai.py --manifest ./output/manifest.json --limit 10
  python ./script/import_openai.py --manifest ./output/manifest.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import ClusterOptions


# -------------------- dotenv (stdlib-only) --------------------
def load_dotenv(path: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file (simple parser, stdlib only)."""
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def env_get(
    name: str, dotenv: dict[str, str], default: str | None = None
) -> str | None:
    """Read env var from process first, then .env, then default."""
    return os.environ.get(name) or dotenv.get(name) or default


# -------------------- config --------------------
@dataclass(frozen=True)
class CouchTarget:
    """Couchbase connection and keyspace settings."""

    connstr: str
    username: str
    password: str
    bucket: str
    scope: str
    collection: str


# -------------------- helpers --------------------
def read_json(path: Path) -> Any:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_doc_id(entry: dict[str, Any]) -> str:
    """
    Stable doc id:
    - use folder name (already collision-safe from parse_openai.py)
    - fallback to index
    """
    folder = entry.get("folder")
    if isinstance(folder, str) and folder.strip():
        return folder.strip()
    idx = entry.get("index")
    return f"conversation_{idx}"


def connect(target: CouchTarget) -> Any:
    """Create a ready Couchbase cluster connection and return (cluster, collection)."""
    auth = PasswordAuthenticator(target.username, target.password)
    cluster = Cluster.connect(target.connstr, ClusterOptions(auth))
    cluster.wait_until_ready(timeout=timedelta(seconds=10))

    bucket = cluster.bucket(target.bucket)
    scope = bucket.scope(target.scope)
    collection = scope.collection(target.collection)
    return cluster, collection


def import_from_manifest(
    manifest_path: Path,
    target: CouchTarget,
    limit: int | None,
    dry_run: bool,
) -> int:
    """Import conversations referenced by manifest.json. Returns number upserted."""
    manifest = read_json(manifest_path)
    conversations = manifest.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("manifest.json missing conversations[] list")

    base_dir = manifest_path.parent
    items = conversations[:limit] if limit else conversations

    _, collection = connect(target)

    print(
        "Import starting\n"
        f"  Manifest : {manifest_path}\n"
        f"  ConnStr  : {target.connstr}\n"
        f"  Keyspace : {target.bucket}.{target.scope}.{target.collection}\n"
        f"  Dry-run  : {'yes' if dry_run else 'no'}\n"
        f"  Items    : {len(items)}"
    )

    upserted = 0
    for i, entry in enumerate(items, start=1):
        if not isinstance(entry, dict):
            continue

        json_path = entry.get("json_path")
        if not isinstance(json_path, str):
            print(
                f"  [{i}/{len(items)}] SKIP invalid json_path", file=sys.stderr
            )
            continue

        p = Path(json_path)
        if not p.is_absolute():
            p = base_dir / p

        if not p.exists():
            print(
                f"  [{i}/{len(items)}] SKIP missing file: {p}", file=sys.stderr
            )
            continue

        convo_obj = read_json(p)
        doc_id = build_doc_id(entry)

        doc = {
            "doc_type": "openai_conversation",
            "manifest": {
                "index": entry.get("index"),
                "title": entry.get("title"),
                "folder": entry.get("folder"),
                "create_time_utc": entry.get("create_time_utc"),
                "update_time_utc": entry.get("update_time_utc"),
                "message_count": entry.get("message_count"),
                "source_json_path": str(p),
            },
            "conversation": convo_obj,
        }

        if dry_run:
            upserted += 1
            if i <= 3 or i == len(items):
                print(f"  [{i}/{len(items)}] DRY upsert id={doc_id}")
            continue

        try:
            collection.upsert(doc_id, doc)
            upserted += 1
            if i <= 3 or i == len(items) or i % 50 == 0:
                print(f"  [{i}/{len(items)}] OK upsert id={doc_id}")
        except CouchbaseException as exc:
            print(
                f"  [{i}/{len(items)}] ERROR id={doc_id}: {exc}",
                file=sys.stderr,
            )

    print(f"Import done\n  Upserted : {upserted}")
    return upserted


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    p = argparse.ArgumentParser(
        description="Import OpenAI archive into Couchbase."
    )
    p.add_argument(
        "--manifest", required=True, help="Path to output/manifest.json"
    )
    p.add_argument(
        "--env-file",
        default=None,
        help="Path to .env (default: ./script/.env)",
    )
    p.add_argument("--connstr", default=None, help="Override CB_CONNSTR")
    p.add_argument("--bucket", default="library")
    p.add_argument("--scope", default="openai")
    p.add_argument("--collection", default="conversations")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    """Main entrypoint."""
    args = build_parser().parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    default_env = Path(__file__).resolve().parent / ".env"
    env_path = (
        Path(args.env_file).expanduser().resolve()
        if args.env_file
        else default_env
    )
    dotenv = load_dotenv(env_path)

    connstr = args.connstr or env_get(
        "CB_CONNSTR", dotenv, "couchbase://127.0.0.1"
    )
    username = env_get("CB_USERNAME", dotenv)
    password = env_get("CB_PASSWORD", dotenv)

    if not username or not password:
        print(
            f"ERROR: missing CB_USERNAME/CB_PASSWORD in {env_path}",
            file=sys.stderr,
        )
        return 2

    target = CouchTarget(
        connstr=connstr,
        username=username,
        password=password,
        bucket=args.bucket,
        scope=args.scope,
        collection=args.collection,
    )

    try:
        import_from_manifest(
            manifest_path=manifest_path,
            target=target,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except CouchbaseException as exc:
        print(f"COUCHBASE ERROR: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
