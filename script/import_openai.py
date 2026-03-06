#!/usr/bin/env python3
# pylint: disable=import-error,too-many-locals,too-many-branches,too-many-statements
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
  python ./script/import_openai.py \
      --manifest ./output/manifest.json --kv-timeout 30
  python ./script/import_openai.py \
      --manifest ./output/manifest.json --skip-existing
  python ./script/import_openai.py \
      --manifest ./output/manifest.json --progress-every 25
  python ./script/import_openai.py \
      --manifest ./output/manifest.json --pause-every 100 --pause-seconds 2
  python ./script/import_openai.py \
      --manifest ./output/manifest.json --verify-bucket-count
  python ./script/import_openai.py \
      --manifest ./output/manifest.json --stats-json ./output/import_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (AmbiguousTimeoutException,
                                  CouchbaseException,
                                  DocumentNotFoundException)
from couchbase.options import ClusterOptions, UpsertOptions

IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


# -------------------- dotenv (stdlib-only) --------------------
def load_dotenv(path: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file."""
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = key.strip()
        v = value.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def env_get(
    name: str,
    dotenv: dict[str, str],
    default: str | None = None,
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


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class ImportOptions:
    """Runtime import behavior."""

    limit: int | None
    dry_run: bool
    kv_timeout_s: int
    max_retries: int
    skip_existing: bool
    progress_every: int
    pause_every: int
    pause_seconds: float
    verify_bucket_count: bool
    stats_json: Path | None


@dataclass
class VerifyResult:
    """Verification results for post-import count checks."""

    enabled: bool = False
    ok: bool = False
    query: str = ""
    collection_count: int | None = None
    expected_selected: int | None = None
    expected_total_manifest: int | None = None
    note: str = ""
    error: str = ""


# pylint: disable=too-many-instance-attributes
@dataclass
class ImportStats:
    """Import statistics."""

    manifest_items_total: int = 0
    items_selected: int = 0
    entries_seen: int = 0
    attempted_upserts: int = 0
    upserted: int = 0
    created: int = 0
    updated: int = 0
    skipped_existing: int = 0
    skipped_invalid_entry: int = 0
    skipped_missing_file: int = 0
    errors: int = 0
    retries: int = 0
    dry_run_items: int = 0
    source_bytes: int = 0
    source_messages: int = 0
    pauses: int = 0
    pause_seconds_total: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    verify: VerifyResult | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Return total elapsed time in seconds."""
        if self.finished_at <= self.started_at:
            return 0.0
        return self.finished_at - self.started_at

    @property
    def active_seconds(self) -> float:
        """Return elapsed time excluding intentional pause time."""
        return max(0.0, self.elapsed_seconds - self.pause_seconds_total)

    @property
    def items_per_second(self) -> float:
        """Return processed selected items per second."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.entries_seen / elapsed

    @property
    def upserts_per_second(self) -> float:
        """Return successful upserts per second."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.upserted / elapsed

    @property
    def active_upserts_per_second(self) -> float:
        """Return successful upserts per active second."""
        active = self.active_seconds
        if active <= 0:
            return 0.0
        return self.upserted / active

    @property
    def mb_read(self) -> float:
        """Return total source bytes read in MB."""
        return self.source_bytes / 1_048_576.0

    @property
    def avg_messages_per_doc(self) -> float:
        """Return average messages per successful upsert."""
        if self.upserted <= 0:
            return 0.0
        return self.source_messages / self.upserted

    @property
    def avg_kb_per_seen_doc(self) -> float:
        """Return average KB read per seen entry."""
        if self.entries_seen <= 0:
            return 0.0
        return (self.source_bytes / 1024.0) / self.entries_seen


# -------------------- helpers --------------------
def validate_identifier(name: str, label: str) -> str:
    """
    Validate Couchbase identifier used in N1QL.

    Only allow simple identifiers consisting of:
    - letters
    - digits
    - underscore
    - hyphen

    Identifiers must start with a letter.
    """
    if not isinstance(name, str) or not IDENTIFIER_RE.fullmatch(name):
        raise ValueError(
            f"Invalid {label}: {name!r}. "
            "Allowed pattern: ^[A-Za-z][A-Za-z0-9_-]*$"
        )
    return name


def read_json(path: Path) -> Any:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_size(path: Path) -> int:
    """Return file size in bytes."""
    return path.stat().st_size


def build_doc_id(entry: dict[str, Any]) -> str:
    """
    Stable doc id:
    - use folder name
    - fallback to index
    """
    folder = entry.get("folder")
    if isinstance(folder, str) and folder.strip():
        return folder.strip()
    idx = entry.get("index")
    return f"conversation_{idx}"


def format_duration(seconds: float) -> str:
    """Format seconds into a readable duration."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def connect(target: CouchTarget) -> Any:
    """Create a ready Couchbase cluster connection and return (cluster, collection)."""
    auth = PasswordAuthenticator(target.username, target.password)
    cluster = Cluster.connect(target.connstr, ClusterOptions(auth))
    cluster.wait_until_ready(timeout=timedelta(seconds=10))

    bucket = cluster.bucket(target.bucket)
    scope = bucket.scope(target.scope)
    collection = scope.collection(target.collection)
    return cluster, collection


def doc_exists(collection: Any, doc_id: str, kv_timeout_s: int) -> bool:
    """Check whether a document already exists."""
    try:
        collection.get(doc_id, timeout=timedelta(seconds=kv_timeout_s))
        return True
    except DocumentNotFoundException:
        return False


# pylint: disable=too-many-arguments,too-many-positional-arguments
def upsert_with_retry(
    collection: Any,
    doc_id: str,
    doc: dict[str, Any],
    kv_timeout_s: int,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> int:
    """
    Upsert a document with explicit KV timeout and exponential backoff retry.

    Returns the number of retries performed for this document.
    """
    opts = UpsertOptions(timeout=timedelta(seconds=kv_timeout_s))
    last_exc: Exception | None = None
    retries_used = 0

    for attempt in range(1, max_retries + 1):
        try:
            collection.upsert(doc_id, doc, opts)
            return retries_used
        except AmbiguousTimeoutException as exc:
            last_exc = exc
            if attempt < max_retries:
                retries_used += 1
                wait = backoff_s * attempt
                print(
                    f"    RETRY {attempt}/{max_retries - 1} timeout, "
                    f"waiting {wait:.1f}s  id={doc_id}",
                    file=sys.stderr,
                )
                time.sleep(wait)

    raise last_exc  # type: ignore[misc]


def count_bucket_documents(
    cluster: Any,
    target: CouchTarget,
) -> tuple[str, int]:
    """
    Count documents using bucket-level Couchbase stats.

    This is exact only when the bucket is dedicated to this dataset.
    """
    bucket_name = validate_identifier(target.bucket, "bucket")
    manager = cluster.buckets()
    bucket_settings = manager.get_bucket(bucket_name)

    item_count = getattr(bucket_settings, "num_items", None)
    if not isinstance(item_count, int):
        raise ValueError(
            f"Could not read bucket item count for {bucket_name!r}."
        )

    return f"bucket_settings.num_items:{bucket_name}", item_count


def count_collection_documents(
    cluster: Any,
    target: CouchTarget,
) -> tuple[str, int]:
    """
    Count documents using bucket-level Couchbase stats.

    This is exact only when the target bucket is dedicated to this dataset.
    """
    bucket_name = validate_identifier(target.bucket, "bucket")
    manager = cluster.buckets()
    settings = manager.get_bucket(bucket_name)

    basic_stats = settings.get("basicStats", {})
    item_count = basic_stats.get("itemCount")

    if not isinstance(item_count, int):
        raise ValueError(
            f"Could not read itemCount for bucket {bucket_name!r}."
        )

    return f"bucket_stats:{bucket_name}", item_count


def print_progress(
    current: int,
    total: int,
    stats: ImportStats,
    action: str,
    doc_id: str,
) -> None:
    """Print a progress block with elapsed time and rates."""
    print(
        f"  [{current}/{total}] {action} id={doc_id}\n"
        f"    Elapsed      : {format_duration(stats.elapsed_seconds)}\n"
        f"    Seen         : {stats.entries_seen:,}/{total:,}\n"
        f"    Upserted     : {stats.upserted:,}\n"
        f"    Created      : {stats.created:,}\n"
        f"    Updated      : {stats.updated:,}\n"
        f"    Skipped exist: {stats.skipped_existing:,}\n"
        f"    Missing file : {stats.skipped_missing_file:,}\n"
        f"    Errors       : {stats.errors:,}\n"
        f"    Retries      : {stats.retries:,}\n"
        f"    Read         : {stats.mb_read:.1f} MB\n"
        f"    Rate         : {stats.items_per_second:.2f} items/s\n"
        f"    Upsert rate  : {stats.upserts_per_second:.2f} docs/s"
    )


def maybe_pause(
    current: int,
    total: int,
    options: ImportOptions,
    stats: ImportStats,
) -> None:
    """Pause processing if configured."""
    if options.pause_every <= 0:
        return
    if current % options.pause_every != 0:
        return
    if current == total:
        return

    stats.pauses += 1
    stats.pause_seconds_total += options.pause_seconds
    print(
        f"    PAUSE after {current:,} items for {options.pause_seconds:.1f}s",
        file=sys.stderr,
    )
    time.sleep(options.pause_seconds)


def should_print_progress(
    current: int,
    total: int,
    options: ImportOptions,
) -> bool:
    """Decide whether to print progress for the current item."""
    if current <= 3 or current == total:
        return True
    return options.progress_every > 0 and current % options.progress_every == 0


def build_document(
    entry: dict[str, Any], source_path: Path, convo_obj: Any
) -> dict[str, Any]:
    """Build Couchbase document payload."""
    return {
        "doc_type": "openai_conversation",
        "manifest": {
            "index": entry.get("index"),
            "title": entry.get("title"),
            "folder": entry.get("folder"),
            "create_time_utc": entry.get("create_time_utc"),
            "update_time_utc": entry.get("update_time_utc"),
            "message_count": entry.get("message_count"),
            "source_json_path": str(source_path),
        },
        "conversation": convo_obj,
    }


def verify_import_counts(
    cluster: Any,
    target: CouchTarget,
    stats: ImportStats,
) -> VerifyResult:
    """Verify bucket count against selected and manifest totals."""
    result = VerifyResult(
        enabled=True,
        expected_selected=stats.items_selected,
        expected_total_manifest=stats.manifest_items_total,
    )

    try:
        statement, bucket_count = count_bucket_documents(cluster, target)
        result.query = statement
        result.collection_count = bucket_count

        if bucket_count == stats.items_selected:
            result.ok = True
            result.note = (
                "Bucket item count matches selected manifest item count. "
                "This is only exact when the bucket is dedicated to this dataset."
            )
        elif bucket_count == stats.manifest_items_total:
            result.ok = True
            result.note = (
                "Bucket item count matches total manifest item count. "
                "This is only exact when the bucket is dedicated to this dataset."
            )
        else:
            result.ok = False
            result.note = (
                "Bucket item count does not match selected or total manifest count. "
                "Bucket-level verification may include unrelated documents."
            )

    except (CouchbaseException, ValueError, AttributeError) as exc:
        result.ok = False
        result.error = str(exc)

    return result


def stats_to_dict(
    stats: ImportStats, target: CouchTarget, options: ImportOptions
) -> dict[str, Any]:
    """Serialize stats and run context to a JSON-friendly dict."""
    payload = asdict(stats)
    payload["derived"] = {
        "elapsed_seconds": stats.elapsed_seconds,
        "active_seconds": stats.active_seconds,
        "items_per_second": stats.items_per_second,
        "upserts_per_second": stats.upserts_per_second,
        "active_upserts_per_second": stats.active_upserts_per_second,
        "mb_read": stats.mb_read,
        "avg_messages_per_doc": stats.avg_messages_per_doc,
        "avg_kb_per_seen_doc": stats.avg_kb_per_seen_doc,
    }
    payload["target"] = {
        "connstr": target.connstr,
        "username": target.username,
        "credentials_redacted": True,
        "bucket": target.bucket,
        "scope": target.scope,
        "collection": target.collection,
    }
    payload["options"] = {
        "limit": options.limit,
        "dry_run": options.dry_run,
        "kv_timeout_s": options.kv_timeout_s,
        "max_retries": options.max_retries,
        "skip_existing": options.skip_existing,
        "progress_every": options.progress_every,
        "pause_every": options.pause_every,
        "pause_seconds": options.pause_seconds,
        "verify_bucket_count": options.verify_bucket_count,
        "stats_json": (
            str(options.stats_json) if options.stats_json else None
        ),
    }
    return payload


def import_from_manifest(
    manifest_path: Path,
    target: CouchTarget,
    options: ImportOptions,
) -> ImportStats:
    """Import conversations referenced by manifest.json."""
    manifest = read_json(manifest_path)
    conversations = manifest.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("manifest.json missing conversations[] list")

    base_dir = manifest_path.parent
    items = conversations[: options.limit] if options.limit else conversations

    cluster, collection = connect(target)

    stats = ImportStats(
        manifest_items_total=len(conversations),
        items_selected=len(items),
        started_at=time.perf_counter(),
    )

    pause_label = (
        str(options.pause_every) if options.pause_every > 0 else "off"
    )
    stats_label = str(options.stats_json) if options.stats_json else "off"

    print(
        "Import starting\n"
        f"  Manifest       : {manifest_path}\n"
        f"  ConnStr        : {target.connstr}\n"
        f"  Keyspace       : {target.bucket}.{target.scope}"
        f".{target.collection}\n"
        f"  Dry-run        : {'yes' if options.dry_run else 'no'}\n"
        f"  Skip existing  : {'yes' if options.skip_existing else 'no'}\n"
        f"  Verify bucket count   : {'yes' if options.verify_bucket_count else 'no'}\n"
        f"  KV timeout     : {options.kv_timeout_s}s\n"
        f"  Max retries    : {options.max_retries}\n"
        f"  Progress every : {options.progress_every}\n"
        f"  Pause every    : {pause_label}\n"
        f"  Pause seconds  : {options.pause_seconds:.1f}\n"
        f"  Stats JSON     : {stats_label}\n"
        f"  Manifest items : {len(conversations)}\n"
        f"  Selected items : {len(items)}"
    )

    total = len(items)

    for i, entry in enumerate(items, start=1):
        stats.entries_seen += 1

        if not isinstance(entry, dict):
            stats.skipped_invalid_entry += 1
            maybe_pause(i, total, options, stats)
            continue

        json_path = entry.get("json_path")
        if not isinstance(json_path, str):
            stats.skipped_invalid_entry += 1
            print(f"  [{i}/{total}] SKIP invalid json_path", file=sys.stderr)
            maybe_pause(i, total, options, stats)
            continue

        source_path = Path(json_path)
        if not source_path.is_absolute():
            source_path = base_dir / source_path

        if not source_path.exists():
            stats.skipped_missing_file += 1
            print(
                f"  [{i}/{total}] SKIP missing file: {source_path}",
                file=sys.stderr,
            )
            maybe_pause(i, total, options, stats)
            continue

        stats.source_bytes += file_size(source_path)

        convo_obj = read_json(source_path)
        doc_id = build_doc_id(entry)

        message_count = entry.get("message_count")
        if isinstance(message_count, int):
            stats.source_messages += message_count

        exists_before = False
        try:
            exists_before = doc_exists(
                collection, doc_id, options.kv_timeout_s
            )
        except CouchbaseException as exc:
            stats.errors += 1
            stats.finished_at = time.perf_counter()
            print(
                f"  [{i}/{total}] ERROR exists-check id={doc_id}: {exc}",
                file=sys.stderr,
            )
            maybe_pause(i, total, options, stats)
            continue

        if options.skip_existing and exists_before:
            stats.skipped_existing += 1
            stats.finished_at = time.perf_counter()
            if should_print_progress(i, total, options):
                print_progress(i, total, stats, "SKIP existing", doc_id)
            maybe_pause(i, total, options, stats)
            continue

        if options.dry_run:
            stats.dry_run_items += 1
            stats.upserted += 1
            if exists_before:
                stats.updated += 1
            else:
                stats.created += 1
            stats.finished_at = time.perf_counter()

            if should_print_progress(i, total, options):
                action = "DRY update" if exists_before else "DRY create"
                print_progress(i, total, stats, action, doc_id)

            maybe_pause(i, total, options, stats)
            continue

        doc = build_document(entry, source_path, convo_obj)

        try:
            stats.attempted_upserts += 1
            retry_count = upsert_with_retry(
                collection=collection,
                doc_id=doc_id,
                doc=doc,
                kv_timeout_s=options.kv_timeout_s,
                max_retries=options.max_retries,
            )
            stats.retries += retry_count
            stats.upserted += 1
            if exists_before:
                stats.updated += 1
            else:
                stats.created += 1
            stats.finished_at = time.perf_counter()

            if should_print_progress(i, total, options):
                action = "OK update" if exists_before else "OK create"
                print_progress(i, total, stats, action, doc_id)

        except AmbiguousTimeoutException as exc:
            stats.errors += 1
            stats.finished_at = time.perf_counter()
            print(
                f"  [{i}/{total}] ERROR timeout after "
                f"{options.max_retries} attempts id={doc_id}: {exc}",
                file=sys.stderr,
            )
        except CouchbaseException as exc:
            stats.errors += 1
            stats.finished_at = time.perf_counter()
            print(
                f"  [{i}/{total}] ERROR id={doc_id}: {exc}",
                file=sys.stderr,
            )

        maybe_pause(i, total, options, stats)

    stats.finished_at = time.perf_counter()

    if options.verify_bucket_count and not options.dry_run:
        stats.verify = verify_import_counts(cluster, target, stats)

    return stats


def print_final_stats(stats: ImportStats) -> None:
    """Print final import statistics."""
    print(
        "\nImport done\n"
        f"  Manifest items   : {stats.manifest_items_total:,}\n"
        f"  Selected items   : {stats.items_selected:,}\n"
        f"  Entries seen     : {stats.entries_seen:,}\n"
        f"  Attempted upserts: {stats.attempted_upserts:,}\n"
        f"  Upserted         : {stats.upserted:,}\n"
        f"  Created          : {stats.created:,}\n"
        f"  Updated          : {stats.updated:,}\n"
        f"  Dry-run items    : {stats.dry_run_items:,}\n"
        f"  Skipped existing : {stats.skipped_existing:,}\n"
        f"  Skipped invalid  : {stats.skipped_invalid_entry:,}\n"
        f"  Missing files    : {stats.skipped_missing_file:,}\n"
        f"  Errors           : {stats.errors:,}\n"
        f"  Retries          : {stats.retries:,}\n"
        f"  Messages seen    : {stats.source_messages:,}\n"
        f"  Data read        : {stats.mb_read:.1f} MB\n"
        f"  Pauses           : {stats.pauses:,}\n"
        f"  Pause time       : {format_duration(stats.pause_seconds_total)}\n"
        f"  Elapsed          : {format_duration(stats.elapsed_seconds)}\n"
        f"  Rate             : {stats.items_per_second:.2f} items/s\n"
        f"  Upsert rate      : {stats.upserts_per_second:.2f} docs/s\n"
        f"  Active time      : {format_duration(stats.active_seconds)}\n"
        f"  Active rate      : {stats.active_upserts_per_second:.2f} docs/s\n"
        f"  Avg msgs/doc     : {stats.avg_messages_per_doc:.2f}\n"
        f"  Avg size/doc     : {stats.avg_kb_per_seen_doc:.1f} KB"
    )

    if stats.verify is not None and stats.verify.enabled:
        print("\nVerify count")
        if stats.verify.error:
            print("  Status           : ERROR")
            print(f"  Error            : {stats.verify.error}")
        else:
            status = "OK" if stats.verify.ok else "MISMATCH"
            print(f"  Status           : {status}")
            print(f"  Bucket item count: {stats.verify.collection_count:,}")
            print(f"  Expected selected: {stats.verify.expected_selected:,}")
            print(
                "  Expected total   : "
                f"{stats.verify.expected_total_manifest:,}"
            )
            print(f"  Note             : {stats.verify.note}")


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Import OpenAI archive into Couchbase."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to output/manifest.json",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env (default: ./script/.env)",
    )
    parser.add_argument("--connstr", default=None, help="Override CB_CONNSTR")
    parser.add_argument("--bucket", default="library")
    parser.add_argument("--scope", default="openai")
    parser.add_argument("--collection", default="conversations")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--kv-timeout",
        type=int,
        default=30,
        metavar="S",
        help="KV upsert timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Max retry attempts on timeout (default: 3)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already exist in Couchbase",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        metavar="N",
        help="Print progress every N items (default: 50)",
    )
    parser.add_argument(
        "--pause-every",
        type=int,
        default=0,
        metavar="N",
        help="Pause every N items, 0 disables pauses (default: 0)",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        metavar="S",
        help="Seconds to pause when --pause-every is triggered (default: 2.0)",
    )
    parser.add_argument(
        "--verify-bucket-count",
        action="store_true",
        help="Compare bucket doc count with manifest counts after import",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Write final run stats to a JSON file",
    )
    return parser


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
        "CB_CONNSTR",
        dotenv,
        "couchbase://127.0.0.1",
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

    stats_json_path = (
        Path(args.stats_json).expanduser().resolve()
        if args.stats_json
        else None
    )

    options = ImportOptions(
        limit=args.limit,
        dry_run=args.dry_run,
        kv_timeout_s=args.kv_timeout,
        max_retries=args.max_retries,
        skip_existing=args.skip_existing,
        progress_every=max(1, args.progress_every),
        pause_every=max(0, args.pause_every),
        pause_seconds=max(0.0, args.pause_seconds),
        verify_bucket_count=args.verify_bucket_count,
        stats_json=stats_json_path,
    )

    try:
        stats = import_from_manifest(
            manifest_path=manifest_path,
            target=target,
            options=options,
        )
        print_final_stats(stats)

        if options.stats_json is not None:
            payload = stats_to_dict(stats, target, options)
            write_json(options.stats_json, payload)
            print(f"\nStats JSON written to: {options.stats_json}")

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except CouchbaseException as exc:
        print(f"COUCHBASE ERROR: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
