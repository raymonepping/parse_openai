#!/usr/bin/env python3
"""
chatgpt_export_unpacker.py  v1.2.3

Turns a ChatGPT data export folder into:
- One big JSON  "as-is"
  -> all_conversations.json
- One big Markdown "as-is" (optional, default on)
  -> all_conversations.md
- One JSON per conversation
  -> conversations/<folder>/conversation.json
- One Markdown per conversation (optional, default on)
  -> conversations/<folder>/conversation.md
- Summary manifest with hashes
  -> manifest.json

Zero external dependencies (stdlib only).
Requires Python 3.10+.

Usage:
  python3 chatgpt_export_unpacker.py \
    --input ~/Downloads/chatgpt-export --output ./archive --hash

  python3 chatgpt_export_unpacker.py \
    --input ~/Downloads/chatgpt-export/conversations.json --output ./archive

Options:
  --stats        Print fun stats after processing
  --no-markdown  Skip all Markdown outputs (faster, smaller)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

# -- version ------------------------------------------------------------------
__version__ = "1.2.3"
MIN_PYTHON = (3, 10)

# -- logging ------------------------------------------------------------------
log = logging.getLogger("unpacker")


# -- utilities ----------------------------------------------------------------
def utc_iso_from_ts(ts: float | None) -> str:
    """Convert a UNIX timestamp to UTC ISO-8601, or 'unknown' on failure."""
    if ts is None:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (ValueError, OSError, OverflowError):
        return "unknown"


def safe_slug(text: str, max_len: int = 80) -> str:
    """Return a filesystem-safe slug truncated to max_len characters."""
    text = (text or "").strip()
    if not text:
        text = "untitled"
    text = re.sub(r"[\/\\\0]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^A-Za-z0-9 ._\-]+", "_", text)
    text = text.strip(" ._-")
    if not text:
        text = "untitled"
    return text[:max_len].rstrip(" ._-") if len(text) > max_len else text


def atomic_write_text(
    path: Path, content: str, encoding: str = "utf-8"
) -> None:
    """Write text to path atomically via a .tmp swap to prevent partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialize data as JSON and write to path atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def stream_append_text(
    path: Path, content: str, encoding: str = "utf-8"
) -> None:
    """Append content directly to path; avoids accumulating large strings in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=encoding) as fh:
        fh.write(content)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json_file(path: Path) -> Any:
    """Load a JSON file with explicit UTF-8 decoding and BOM tolerance."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return json.loads(raw.decode("utf-8"))


def _word_count(text: str) -> int:
    """Fast word count for stats."""
    return len(text.split())


# -- data model ---------------------------------------------------------------
@dataclass(frozen=True)
class MessageRecord:
    """Immutable representation of a single linearized chat message."""

    role: str
    created: float | None
    text: str


@dataclass
class TimeBounds:
    """Track earliest and latest message timestamps."""

    earliest_ts: float | None = None
    latest_ts: float | None = None

    def update(self, ts: float | None) -> None:
        """Update bounds with a single timestamp."""
        if ts is None:
            return
        if self.earliest_ts is None or ts < self.earliest_ts:
            self.earliest_ts = ts
        if self.latest_ts is None or ts > self.latest_ts:
            self.latest_ts = ts


@dataclass
class LongestConversation:
    """Track the conversation with the most messages."""

    messages: int = 0
    title: str = ""
    folder: str = ""

    def update(self, count: int, title: str, folder: str) -> None:
        """Update the tracked longest conversation if count is larger."""
        if count > self.messages:
            self.messages = count
            self.title = title
            self.folder = folder


@dataclass
class Totals:
    """Aggregate totals for the archive."""

    conversations: int = 0
    messages: int = 0
    chars: int = 0
    words: int = 0


@dataclass
class Stats:
    """Collect processing stats without a second pass."""

    totals: Totals = field(default_factory=Totals)
    role_counts: Counter[str] = field(default_factory=Counter)
    bounds: TimeBounds = field(default_factory=TimeBounds)
    longest: LongestConversation = field(default_factory=LongestConversation)


@dataclass
class ConversationContext:
    """Shared context for per-conversation processing."""

    convos_dir: Path
    big_md_out: Path | None
    used_folders: set[str]
    include_system: bool
    do_hash: bool
    write_markdown: bool
    stats: Stats | None


@dataclass(frozen=True)
class ArchiveBuildConfig:
    """Configuration snapshot used to build manifest and outputs."""

    conv_json_path: Path
    output_dir: Path
    now_utc: str
    total_conversations: int
    processed_conversations: int
    big_json_out: Path
    big_md_out: Path | None


def extract_message_text(message_obj: dict[str, Any]) -> str:
    """Extract plain text from a ChatGPT message object, handling known variants."""
    content = message_obj.get("content") or {}
    parts = content.get("parts")
    if isinstance(parts, list):
        return "\n".join([p for p in parts if isinstance(p, str)]).strip()
    if isinstance(content.get("text"), str):
        return content["text"].strip()
    return ""


def is_user_visible_role(role: str, include_system: bool) -> bool:
    """Return True if the role should appear in Markdown output."""
    if role in ("user", "assistant"):
        return True
    return include_system and role == "system"


def _child_score(node: dict[str, Any]) -> tuple[float, str]:
    """Score a child node for branch selection: prefer most-recent, then lexical."""
    msg = node.get("message") or {}
    ct = msg.get("create_time")
    ct_f = float(ct) if isinstance(ct, (int, float)) else -1.0
    return (ct_f, node.get("id") or "")


# -- graph helpers ------------------------------------------------------------
def _build_mapping_graph(
    mapping: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str | None],
    dict[str, list[str]],
]:
    """Parse the message mapping into nodes, parent, and children dicts."""
    nodes: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str | None] = {}
    children_of: dict[str, list[str]] = {}

    for node_id, node in mapping.items():
        if not isinstance(node, dict):
            continue
        nodes[node_id] = node
        parent = node.get("parent")
        parent_of[node_id] = (
            parent if isinstance(parent, str) or parent is None else None
        )
        children = node.get("children") or []
        if isinstance(children, list):
            children_of[node_id] = [c for c in children if isinstance(c, str)]
        else:
            children_of[node_id] = []

    return nodes, parent_of, children_of


def _node_to_record(
    nid: str,
    nodes: dict[str, dict[str, Any]],
    include_system: bool,
) -> MessageRecord | None:
    """Convert a single graph node to a MessageRecord, or None if not renderable."""
    node = nodes.get(nid) or {}
    msg = node.get("message")
    if not isinstance(msg, dict):
        return None

    author = msg.get("author") or {}
    role = (author.get("role") or "unknown").strip()
    if not is_user_visible_role(role, include_system):
        return None

    created = msg.get("create_time")
    created_f = float(created) if isinstance(created, (int, float)) else None
    text = extract_message_text(msg)

    if not text and role not in ("system",):
        return None

    return MessageRecord(role=role, created=created_f, text=text)


def _follow_main_path(
    root_id: str,
    nodes: dict[str, dict[str, Any]],
    children_of: dict[str, list[str]],
    include_system: bool,
) -> list[MessageRecord]:
    """Walk the main (most-recent) branch of the conversation graph."""
    transcript: list[MessageRecord] = []
    visited: set[str] = set()
    current: str | None = root_id

    while current and current not in visited:
        visited.add(current)
        rec = _node_to_record(current, nodes, include_system)
        if rec is not None:
            transcript.append(rec)

        kids = children_of.get(current, [])
        if not kids:
            break

        current = sorted(
            kids,
            key=lambda cid: _child_score(nodes.get(cid) or {}),
            reverse=True,
        )[0]

    return transcript


def linearize_conversation(
    convo: dict[str, Any],
    include_system: bool,
) -> list[MessageRecord]:
    """Reconstruct a linear transcript from a ChatGPT conversation graph."""
    mapping = convo.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return []

    nodes, parent_of, children_of = _build_mapping_graph(mapping)
    roots = [nid for nid, p in parent_of.items() if p is None]
    root_id = roots[0] if roots else None

    if root_id:
        transcript = _follow_main_path(
            root_id, nodes, children_of, include_system
        )
        if transcript:
            return transcript

    all_recs: list[MessageRecord] = []
    for nid in nodes:
        rec = _node_to_record(nid, nodes, include_system)
        if rec is not None:
            all_recs.append(rec)

    all_recs.sort(key=lambda r: (r.created is None, r.created or 0.0))
    return all_recs


def render_markdown_from_transcript(
    title: str,
    created_iso: str,
    updated_iso: str,
    transcript: list[MessageRecord],
) -> str:
    """Render Markdown from an already-linearized transcript."""
    lines: list[str] = [
        f"# {title}".rstrip(),
        "",
        f"- Created (UTC): {created_iso}",
        f"- Updated (UTC): {updated_iso}",
        "",
    ]

    if not transcript:
        lines.extend(["_No messages found in this conversation._", ""])
        return "\n".join(lines)

    role_label = {"user": "User", "assistant": "Assistant"}
    for rec in transcript:
        header = role_label.get(rec.role, rec.role.capitalize())
        ts = utc_iso_from_ts(rec.created)
        lines.extend([f"## {header} ({ts})", "", rec.text.rstrip(), ""])

    return "\n".join(lines)


# -- collision-safe folder naming ---------------------------------------------
def unique_folder(
    base_dir: Path, name: str, used: set[str], index: int
) -> Path:
    """Return a folder path guaranteed not to collide with already-used names."""
    candidate = safe_slug(name)
    if candidate not in used:
        used.add(candidate)
        return base_dir / candidate
    suffixed = safe_slug(f"{name}_{index:04d}")
    used.add(suffixed)
    return base_dir / suffixed


# -- memories -----------------------------------------------------------------
def _render_memories_md(source: Path, memories: list[Any]) -> str:
    """Render a memories list as a Markdown document."""
    lines: list[str] = ["# ChatGPT Memories", "", f"- Source: `{source}`", ""]
    for i, mem in enumerate(memories, start=1):
        if isinstance(mem, dict):
            text = mem.get("text") or mem.get("content") or json.dumps(mem)
            created = utc_iso_from_ts(
                mem.get("created_at") or mem.get("timestamp")
            )
            lines.extend(
                [f"### Memory {i} ({created})", "", str(text).strip(), ""]
            )
        else:
            lines.extend([f"### Memory {i}", "", str(mem).strip(), ""])
    return "\n".join(lines)


def process_memories(
    input_dir: Path,
    output_dir: Path,
    do_hash: bool,
    write_markdown: bool,
) -> dict[str, Any] | None:
    """Locate and export memories from the export folder."""
    candidates = [input_dir / "memories.json", input_dir / "memory.json"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        return None

    log.info("Found memories: %s", source)
    data = load_json_file(source)
    memories = data if isinstance(data, list) else data.get("memories", [])

    json_out = output_dir / "memories.json"
    atomic_write_json(json_out, data)

    md_out: Path | None = None
    if write_markdown:
        md_out = output_dir / "memories.md"
        atomic_write_text(md_out, _render_memories_md(source, memories))

    entry: dict[str, Any] = {
        "source": str(source),
        "memory_count": len(memories),
        "json_path": str(json_out),
    }
    if md_out is not None:
        entry["md_path"] = str(md_out)

    if do_hash:
        sha: dict[str, str] = {"json": sha256_file(json_out)}
        if md_out is not None:
            sha["md"] = sha256_file(md_out)
        entry["sha256"] = sha

    return entry


# -- input validation ---------------------------------------------------------
def resolve_conversations_json(input_path: Path) -> tuple[Path, Path]:
    """Locate conversations.json; return (file path, parent export directory)."""
    if input_path.is_file() and input_path.name == "conversations.json":
        return input_path, input_path.parent
    if input_path.is_dir():
        candidate = input_path / "conversations.json"
        if candidate.exists():
            return candidate, input_path
    raise FileNotFoundError(
        f"Could not find conversations.json at: {input_path}"
    )


def validate_export_shape(data: Any) -> list[dict[str, Any]]:
    """Validate that the loaded JSON is a non-empty list of conversation objects."""
    if not isinstance(data, list):
        raise ValueError("Expected conversations.json to be a JSON array.")
    convos = [c for c in data if isinstance(c, dict)]
    if not convos:
        raise ValueError(
            "No conversation objects found in conversations.json."
        )
    return convos


def iter_conversations(
    conversations: list[dict[str, Any]],
    limit: int | None,
) -> Generator[tuple[int, dict[str, Any]], None, None]:
    """Yield (1-based index, conversation) pairs, optionally limited to first N."""
    items = conversations[:limit] if limit else conversations
    yield from enumerate(items, start=1)


# -- main sub-tasks -----------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Unpack a ChatGPT conversations.json into per-chat JSON/MD "
            "plus aggregated outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print archive statistics after processing",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip generating Markdown outputs (per-conversation and aggregated)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to export folder or conversations.json",
    )
    parser.add_argument(
        "--export-memories",
        action="store_true",
        help="If memories.json exists in the export folder, export it too",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include system-role messages in Markdown",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA256 hashes and include in manifest",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        metavar="N",
        help="Limit processing to first N conversations (useful for testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _init_big_md(path: Path, source: Path, now_utc: str, count: int) -> None:
    """Write the header block of the aggregated Markdown file."""
    path.unlink(missing_ok=True)
    header = "\n".join(
        [
            "# ChatGPT Export Archive",
            "",
            f"- Source: `{source}`",
            f"- Generated (UTC): {now_utc}",
            f"- Conversations: {count}",
            "",
        ]
    )
    stream_append_text(path, header)


def _update_stats_for_conversation(
    stats: Stats,
    title: str,
    folder_name: str,
    transcript: list[MessageRecord],
) -> None:
    """Update Stats with one conversation transcript."""
    stats.totals.conversations += 1
    stats.totals.messages += len(transcript)

    for rec in transcript:
        stats.role_counts[rec.role] += 1
        stats.totals.chars += len(rec.text)
        stats.totals.words += _word_count(rec.text)
        stats.bounds.update(rec.created)

    stats.longest.update(len(transcript), title, folder_name)


def process_one_conversation(
    idx: int,
    convo: dict[str, Any],
    ctx: ConversationContext,
) -> dict[str, Any]:
    """Write JSON (and optional MD) for one conversation; return manifest entry."""
    title = convo.get("title") or "untitled"
    created_iso = utc_iso_from_ts(convo.get("create_time"))
    updated_iso = utc_iso_from_ts(convo.get("update_time"))

    date_prefix = (
        created_iso.split("T")[0] if "T" in created_iso else "unknown-date"
    )
    raw_name = safe_slug(f"{date_prefix}_{title}")
    folder = unique_folder(ctx.convos_dir, raw_name, ctx.used_folders, idx)

    json_path = folder / "conversation.json"
    atomic_write_json(json_path, convo)

    transcript = linearize_conversation(
        convo, include_system=ctx.include_system
    )

    if ctx.stats is not None:
        _update_stats_for_conversation(
            stats=ctx.stats,
            title=title,
            folder_name=folder.name,
            transcript=transcript,
        )

    md_path: Path | None = None
    if ctx.write_markdown:
        md_path = folder / "conversation.md"
        md = render_markdown_from_transcript(
            title=title,
            created_iso=created_iso,
            updated_iso=updated_iso,
            transcript=transcript,
        )
        atomic_write_text(md_path, md)

        if ctx.big_md_out is not None:
            stream_append_text(ctx.big_md_out, md)
            stream_append_text(ctx.big_md_out, "\n\n---\n\n")

    entry: dict[str, Any] = {
        "index": idx,
        "title": title,
        "create_time_utc": created_iso,
        "update_time_utc": updated_iso,
        "message_count": len(transcript),
        "folder": folder.name,
        "json_path": str(json_path),
    }
    if md_path is not None:
        entry["md_path"] = str(md_path)

    if ctx.do_hash:
        sha: dict[str, str] = {"json": sha256_file(json_path)}
        if md_path is not None:
            sha["md"] = sha256_file(md_path)
        entry["sha256"] = sha

    return entry


def load_conversations_from_input(
    input_path: Path,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Resolve conversations.json and return (path, export_dir, conversations)."""
    conv_json_path, export_dir = resolve_conversations_json(input_path)
    raw_data = load_json_file(conv_json_path)
    conversations = validate_export_shape(raw_data)
    return conv_json_path, export_dir, conversations


def write_big_outputs(
    output_dir: Path,
    conv_json_path: Path,
    conversations: list[dict[str, Any]],
    limit: int | None,
    write_markdown: bool,
) -> ArchiveBuildConfig:
    """Write aggregated outputs and return config needed to build a manifest."""
    total = len(conversations)
    processed = min(limit, total) if limit else total

    big_json_out = output_dir / "all_conversations.json"
    atomic_write_json(
        big_json_out, conversations[:limit] if limit else conversations
    )

    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    big_md_out: Path | None = None
    if write_markdown:
        big_md_out = output_dir / "all_conversations.md"
        _init_big_md(big_md_out, conv_json_path, now_utc, processed)

    return ArchiveBuildConfig(
        conv_json_path=conv_json_path,
        output_dir=output_dir,
        now_utc=now_utc,
        total_conversations=total,
        processed_conversations=processed,
        big_json_out=big_json_out,
        big_md_out=big_md_out,
    )


def build_manifest_base(cfg: ArchiveBuildConfig) -> dict[str, Any]:
    """Create the initial manifest structure."""
    outputs: dict[str, str] = {
        "all_conversations_json": str(cfg.big_json_out),
        "conversations_dir": str(cfg.output_dir / "conversations"),
    }
    if cfg.big_md_out is not None:
        outputs["all_conversations_md"] = str(cfg.big_md_out)

    return {
        "version": __version__,
        "source": str(cfg.conv_json_path),
        "generated_at_utc": cfg.now_utc,
        "conversation_count_total": cfg.total_conversations,
        "conversation_count_processed": cfg.processed_conversations,
        "outputs": outputs,
        "conversations": [],
    }


def finalize_manifest(
    manifest: dict[str, Any],
    cfg: ArchiveBuildConfig,
    do_hash: bool,
) -> Path:
    """Write manifest.json and return the path."""
    if do_hash:
        sha: dict[str, str] = {
            "all_conversations_json": sha256_file(cfg.big_json_out)
        }
        if cfg.big_md_out is not None:
            sha["all_conversations_md"] = sha256_file(cfg.big_md_out)
        manifest["sha256"] = sha

    manifest_out = cfg.output_dir / "manifest.json"
    atomic_write_json(manifest_out, manifest)
    return manifest_out


def process_all_conversations(
    conversations: list[dict[str, Any]],
    max_conversations: int | None,
    ctx: ConversationContext,
    processed_count: int,
) -> tuple[list[dict[str, Any]], int]:
    """Process conversations and return (manifest_entries, total_message_count)."""
    entries: list[dict[str, Any]] = []
    total_messages = 0

    for idx, convo in iter_conversations(conversations, max_conversations):
        entry = process_one_conversation(idx, convo, ctx)
        total_messages += entry["message_count"]
        entries.append(entry)

        if idx % 25 == 0:
            log.info("  Processed %d/%d...", idx, processed_count)

    return entries, total_messages


def _print_summary(
    process_count: int,
    total_messages: int,
    memories_entry: dict[str, Any] | None,
    output_dir: Path,
    export_memories: bool,
) -> None:
    """Print the final human-readable summary to stdout."""
    mem_label = (
        "yes"
        if memories_entry
        else ("skipped" if not export_memories else "no")
    )

    print(
        "\n Done"
        f"\n  Conversations : {process_count}"
        f"\n  Messages      : {total_messages}"
        f"\n  Memories      : {mem_label}"
        f"\n  Output        : {output_dir}"
    )


def _print_stats(stats: Stats) -> None:
    """Print statistics collected during processing."""
    earliest = utc_iso_from_ts(stats.bounds.earliest_ts)
    latest = utc_iso_from_ts(stats.bounds.latest_ts)
    convos = stats.totals.conversations
    avg = (stats.totals.messages / convos) if convos else 0.0

    print("\n Stats")
    print(f"  Conversations           : {stats.totals.conversations}")
    print(f"  Messages                : {stats.totals.messages}")
    print(f"  Avg msgs/conversation   : {avg:.2f}")
    print(f"  Roles                   : {dict(stats.role_counts)}")
    print(f"  Characters              : {stats.totals.chars}")
    print(f"  Words                   : {stats.totals.words}")
    print(f"  Earliest message (UTC)  : {earliest}")
    print(f"  Latest message (UTC)    : {latest}")
    print("  Longest conversation")
    print(f"    Messages              : {stats.longest.messages}")
    print(f"    Title                 : {stats.longest.title}")
    print(f"    Folder                : {stats.longest.folder}")


def main() -> int:
    """Parse arguments, orchestrate export unpacking, return exit code."""
    if sys.version_info < MIN_PYTHON:
        print(
            f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
            f"You have {sys.version}.",
            file=sys.stderr,
        )
        return 2

    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        conv_json_path, export_dir, conversations = (
            load_conversations_from_input(input_path)
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        log.error("Failed to load conversations.json: %s", exc)
        return 2

    write_markdown = not args.no_markdown
    cfg = write_big_outputs(
        output_dir=output_dir,
        conv_json_path=conv_json_path,
        conversations=conversations,
        limit=args.max_conversations,
        write_markdown=write_markdown,
    )
    manifest = build_manifest_base(cfg)

    stats_obj = Stats() if args.stats else None

    ctx = ConversationContext(
        convos_dir=output_dir / "conversations",
        big_md_out=cfg.big_md_out,
        used_folders=set(),
        include_system=args.include_system,
        do_hash=args.hash,
        write_markdown=write_markdown,
        stats=stats_obj,
    )

    entries, total_messages = process_all_conversations(
        conversations=conversations,
        max_conversations=args.max_conversations,
        ctx=ctx,
        processed_count=cfg.processed_conversations,
    )
    manifest["conversations"] = entries

    memories_entry: dict[str, Any] | None = None
    if args.export_memories:
        memories_entry = process_memories(
            input_dir=export_dir,
            output_dir=output_dir,
            do_hash=args.hash,
            write_markdown=write_markdown,
        )
        if memories_entry:
            manifest["memories"] = memories_entry

    finalize_manifest(manifest, cfg, do_hash=args.hash)
    _print_summary(
        cfg.processed_conversations,
        total_messages,
        memories_entry,
        output_dir,
        export_memories=args.export_memories,
    )

    if stats_obj is not None:
        _print_stats(stats_obj)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
