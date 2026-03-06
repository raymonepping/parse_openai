#!/usr/bin/env python3
"""
generate_conversations.py  v1.4.0

Generates a synthetic OpenAI conversations.json export that matches the
real export schema closely, so your existing parse/unpack scripts work unchanged.

Readable output:
- Default: pretty JSON array with indentation
- Optional: compact JSON array (--compact)
- Optional: NDJSON (--ndjson) for line-by-line inspection

Scales safely:
- Streams output to disk (constant memory) by default
- Progress reporting

Code injection:
- --inject-code          (flag, default: off)
- --code-percent N       (default: 25) percent of assistant messages that get code blocks
- --code-types terraform,hcl,yaml,bash (default: all)

Usage:
  python3 generate_conversations.py --count 100000
  python3 generate_conversations.py --count 1000 --output ./output/conversations.json
  python3 generate_conversations.py --count 500 --seed 42
  python3 generate_conversations.py --count 20000 --ndjson --output conversations.ndjson
  python3 generate_conversations.py --count 50000 --compact
  python3 generate_conversations.py --count 20000 --inject-code --code-percent 30
  python3 generate_conversations.py --count 10000 --inject-code --code-types python,bash
  python3 generate_conversations.py --version

Notes:
- Output is a top-level JSON array by default, matching OpenAI exports:
    [ {conversation}, {conversation}, ... ]
- NDJSON mode outputs one JSON object per line (not an array).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "1.4.0"

# ---------------------------------------------------------------------------
# Topic corpus, weighted toward DevOps / HashiCorp / infra
# ---------------------------------------------------------------------------
TOPICS = [
    # HashiCorp
    "Vault PKI rotation",
    "Vault AppRole auth",
    "Vault dynamic secrets",
    "Vault agent sidecar",
    "Terraform module structure",
    "Terraform remote state",
    "Terraform workspace strategy",
    "Terraform HCP integration",
    "Nomad job spec",
    "Nomad resource allocation",
    "Consul service mesh",
    "Consul health checks",
    "Boundary worker configuration",
    "Boundary OIDC provider",
    "Packer image pipeline",
    # Cloud / infra
    "Kubernetes RBAC",
    "Kubernetes network policy",
    "Docker multi-stage build",
    "Podman rootless setup",
    "GitHub Actions pipeline",
    "ArgoCD app of apps",
    "Helm chart values",
    "Ingress controller config",
    "cert-manager setup",
    "Prometheus alerting rules",
    "Grafana dashboard JSON",
    "Loki log aggregation",
    # Scripting / dev
    "Bash script refactor",
    "Node.js Express middleware",
    "Nuxt 3 composable",
    "Couchbase N1QL query",
    "Couchbase index design",
    "REST API design",
    "OpenAPI spec generation",
    "JWT authentication flow",
    "OAuth2 PKCE flow",
    # General engineering
    "incident post-mortem template",
    "runbook automation",
    "cost optimisation strategy",
    "SLO definition",
    "capacity planning",
    "zero-trust architecture",
    "secrets management strategy",
    "GitOps workflow design",
    "platform engineering roadmap",
    "presales demo environment",
    "technical discovery questions",
    "architecture review board",
    "CAP theorem trade-offs",
]

QUALIFIERS = [
    "deep dive",
    "quick question",
    "experiment",
    "refactor",
    "debug session",
    "implementation plan",
    "review",
    "proof of concept",
    "comparison",
    "best practices",
    "troubleshooting",
    "design discussion",
    "optimisation",
]

USER_MESSAGES = [
    "Can you walk me through how this works in practice?",
    "What are the trade-offs I should be aware of?",
    "Show me a minimal working example.",
    "How would you structure this for production?",
    "What could go wrong here and how do I guard against it?",
    "Is there a cleaner way to do this?",
    "How does this interact with {topic}?",
    "What would you do differently if you were starting from scratch?",
    "Can you help me debug this?",
    "What are the security implications here?",
    "How does this scale beyond a single node?",
    "Give me a checklist for this.",
    "What does the official documentation say about this edge case?",
    "Can you write a script that does this automatically?",
    "How would I test this properly?",
]

ASSISTANT_MESSAGES = [
    "Let me break this down into the key components.",
    "There are a few approaches here, each with different trade-offs.",
    "The short answer is yes, but the details matter. Here is what you need to know.",
    "This is a common pattern. The key is separating concerns early.",
    "I would approach this in three stages: design, implement, validate.",
    "Let me show you a minimal example first, then we can extend it.",
    "The risk here is subtle but real. Here is how to guard against it.",
    "This depends on your consistency requirements. Let me explain why.",
    "Here is a production-grade version with error handling included.",
    "Official guidance helps, but here is the missing bit that matters in practice.",
    "I would restructure this. Here is the reasoning.",
    "This is where many teams run into problems. Here is why.",
    "I can write that script. It will need a couple of inputs first.",
    "The test strategy should cover three scenarios at minimum.",
    "Performance-wise, this holds until you cross a specific threshold.",
]

SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Config dataclass — replaces long param chains across make_conversation
# and both write_*_stream functions
# ---------------------------------------------------------------------------
@dataclass
class ConvConfig:
    start_year:   int
    end_year:     int
    min_turns:    int
    max_turns:    int
    inject_code:  bool
    code_percent: int
    code_types:   Tuple[str, ...]   # tuple so it is hashable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rand_timestamp(start_year: int, end_year: int) -> float:
    """Return a random Unix timestamp between start_year and end_year inclusive."""
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp()
    end   = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    return random.uniform(start, end)


def make_title() -> str:
    return f"{random.choice(TOPICS)} | {random.choice(QUALIFIERS)}"


def make_message_node(
    node_id:   str,
    parent_id: Optional[str],
    children:  List[str],
    role:      str,
    text:      str,
    timestamp: float,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "message": {
            "id":          node_id,
            "author":      {"role": role, "metadata": {}},
            "content":     {"content_type": "text", "parts": [text]},
            "create_time": timestamp,
            "update_time": timestamp,
            "status":      "finished_successfully",
            "metadata":    {},
            "weight":      1.0,
            "recipient":   "all",
        },
        "parent":   parent_id,
        "children": children,
    }


def _parse_code_types(raw: str) -> Tuple[str, ...]:
    allowed = {"terraform", "hcl", "yaml", "bash", "python"}
    parts   = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    out     = [p for p in parts if p in allowed]
    return tuple(sorted(out)) if out else tuple(sorted(allowed))


def _pick_lang(code_types: Tuple[str, ...]) -> str:
    return random.choice(list(code_types)) if code_types else "bash"


def _fence(lang: str) -> str:
    return {"terraform": "hcl", "hcl": "hcl", "yaml": "yaml", "bash": "bash", "python": "python"}.get(lang, "")


@lru_cache(maxsize=64)
def _code_snippets_for(topic: str) -> Dict[str, List[str]]:
    """
    Small library of realistic-ish snippets keyed by language.
    Cached per topic string — lists are built only once per unique topic.
    """
    terraform = [
        """resource "random_id" "suffix" {
  byte_length = 2
}

resource "aws_s3_bucket" "logs" {
  bucket        = "demo-logs-${random_id.suffix.hex}"
  force_destroy = true
}

output "logs_bucket" {
  value = aws_s3_bucket.logs.id
}""",
        """terraform {
  required_version = ">= 1.5.0"
}

locals {
  env = "dev"
}

module "network" {
  source = "./modules/network"
  env    = local.env
}""",
        """data "terraform_remote_state" "core" {
  backend = "local"
  config = {
    path = "../core/terraform.tfstate"
  }
}

output "vpc_id" {
  value = data.terraform_remote_state.core.outputs.vpc_id
}""",
    ]

    hcl = [
        """job "api" {
  datacenters = ["dc1"]

  group "app" {
    network {
      port "http" { to = 3000 }
    }

    task "api" {
      driver = "docker"
      config {
        image = "ghcr.io/acme/api:1.2.3"
        ports = ["http"]
      }

      resources {
        cpu    = 200
        memory = 256
      }
    }
  }
}""",
        """listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = true
}

storage "raft" {
  path = "/vault/file"
}""",
        """worker {
  auth_storage_path = "/var/lib/boundary/worker"
  tags {
    type = ["local"]
    env  = ["dev"]
  }
}""",
    ]

    yaml = [
        """apiVersion: v1
kind: ServiceAccount
metadata:
  name: app
  namespace: demo""",
        """apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: ghcr.io/acme/api:1.2.3
          ports:
            - containerPort: 3000""",
        """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress""",
    ]

    bash = [
        """set -euo pipefail

export VAULT_ADDR="http://127.0.0.1:8200"
vault status || true
vault auth enable approle || true
vault write auth/approle/role/demo token_ttl=1h token_max_ttl=4h""",
        """set -euo pipefail

for svc in vault consul nomad; do
  echo "Checking $svc..."
  curl -fsS "http://127.0.0.1:8500/v1/status/leader" >/dev/null || true
done""",
        """set -euo pipefail

docker build -t demo/app:local .
docker run --rm -p 3000:3000 demo/app:local""",
    ]

    python = [
        """import hvac

client = hvac.Client(url="http://127.0.0.1:8200", token="root")
secret = client.secrets.kv.v2.read_secret_version(
    path="demo/config",
    mount_point="secret",
)
print(secret["data"]["data"])""",
        """import requests

def get_consul_services(host: str = "127.0.0.1", port: int = 8500) -> dict:
    resp = requests.get(f"http://{host}:{port}/v1/catalog/services", timeout=5)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    services = get_consul_services()
    for name, tags in services.items():
        print(f"{name}: {tags}")""",
        """import json
import pathlib

def load_manifest(path: str = "output/manifest.json") -> list[dict]:
    data = pathlib.Path(path).read_text(encoding="utf-8")
    return json.loads(data)

manifest = load_manifest()
for entry in manifest:
    print(f"[{entry['created'][:10]}] {entry['title']}  ({entry['message_count']} messages)")""",
        """from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions

cluster = Cluster(
    "couchbase://localhost",
    ClusterOptions(PasswordAuthenticator("Administrator", "password")),
)
bucket     = cluster.bucket("library")
collection = bucket.scope("openai").collection("conversations")

result = collection.get("2024-02-27_Vault PKI rotation experiment")
print(result.content_as[dict])""",
    ]

    # Light topic bias — extend pools for closely related topics
    t = topic.lower()
    if "terraform" in t:
        terraform.append(
            """variable "env" { type = string }

output "env" {
  value = var.env
}"""
        )
    if any(k in t for k in ("nomad", "boundary", "vault")):
        hcl.append(
            """path "secret/data/demo/*" {
  capabilities = ["read"]
}"""
        )
    if any(k in t for k in ("kubernetes", "openshift", "rbac")):
        yaml.append(
            """apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: read-secrets
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]"""
        )
    if any(k in t for k in ("python", "script", "api", "couchbase", "rest")):
        python.append(
            """import os
import json
import pathlib

OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "output"))

def iter_conversations(output_dir: pathlib.Path):
    for folder in sorted(output_dir.glob("conversations/*")):
        json_file = folder / "conversation.json"
        if json_file.exists():
            yield json.loads(json_file.read_text(encoding="utf-8"))

for conv in iter_conversations(OUTPUT_DIR):
    print(f"{conv['created'][:10]}  {conv['title']}  ({conv['message_count']} messages)")"""
        )

    return {"terraform": terraform, "hcl": hcl, "yaml": yaml, "bash": bash, "python": python}


def maybe_inject_code(
    base_text: str,
    topic:     str,
    cfg:       ConvConfig,
) -> Tuple[str, bool]:
    """
    With probability cfg.code_percent%, append a fenced code block.
    Returns (final_text, injected: bool).
    """
    if not cfg.inject_code or random.randint(1, 100) > max(0, min(100, cfg.code_percent)):
        return base_text, False

    lang       = _pick_lang(cfg.code_types)
    snippets   = _code_snippets_for(topic)
    pool       = snippets.get(lang) or snippets["bash"]
    code       = random.choice(pool)
    fence_lang = _fence(lang)

    return (
        f"{base_text}\n\nHere is a concrete example:\n\n"
        f"```{fence_lang}\n{code}\n```"
    ), True


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------
def make_conversation(
    create_ts: float,
    cfg:       ConvConfig,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Build a single synthetic conversation matching the export schema.
    Returns (conversation_dict, injected_count, assistant_count).
    """
    conv_id       = str(uuid.uuid4())
    title         = make_title()
    turns         = random.randint(cfg.min_turns, cfg.max_turns)
    root_id       = str(uuid.uuid4())
    first_user_id = str(uuid.uuid4())
    mapping: Dict[str, Any] = {}

    mapping[root_id] = make_message_node(
        node_id=root_id, parent_id=None, children=[first_user_id],
        role="system", text=SYSTEM_PROMPT, timestamp=create_ts,
    )

    prev_id         = root_id
    next_user_id    = first_user_id
    last_ts         = create_ts
    injected        = 0
    assistant_count = 0

    for turn in range(turns):
        user_id      = next_user_id
        assistant_id = str(uuid.uuid4())
        next_user_id = str(uuid.uuid4()) if turn < turns - 1 else ""

        base    = create_ts + (turn * random.uniform(45, 180))
        user_ts = max(last_ts + random.uniform(1, 30), base)
        asst_ts = user_ts + random.uniform(2, 20)

        user_text = random.choice(USER_MESSAGES).replace("{topic}", title)
        asst_base = random.choice(ASSISTANT_MESSAGES)
        asst_text, got_code = maybe_inject_code(asst_base, title, cfg)

        assistant_count += 1
        if got_code:
            injected += 1

        mapping[user_id] = make_message_node(
            node_id=user_id, parent_id=prev_id, children=[assistant_id],
            role="user", text=user_text, timestamp=user_ts,
        )
        mapping[assistant_id] = make_message_node(
            node_id=assistant_id, parent_id=user_id,
            children=[next_user_id] if next_user_id else [],
            role="assistant", text=asst_text, timestamp=asst_ts,
        )

        prev_id = assistant_id
        last_ts = asst_ts

    return {
        "id":          conv_id,
        "title":       title,
        "create_time": create_ts,
        "update_time": last_ts,
        "mapping":     mapping,
    }, injected, assistant_count


# ---------------------------------------------------------------------------
# Shared progress line
# ---------------------------------------------------------------------------
def _progress(i: int, count: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate    = (i + 1) / elapsed if elapsed > 0 else 0.0
    print(
        f"  {i+1:,} / {count:,}  ({(i+1)/count*100:5.1f}%)  {rate:,.0f} conv/s",
        end="\r", flush=True,
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_json_array_stream(
    out_path:     Path,
    count:        int,
    pretty:       bool,
    compact:      bool,
    report_every: int,
    cfg:          ConvConfig,
) -> Tuple[int, int]:
    """Write a JSON array to disk, streaming one conversation at a time.
    Returns (total_injected, total_assistant)."""
    indent          = 2 if pretty else None
    separators      = (",", ":") if compact else None
    total_injected  = 0
    total_assistant = 0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        f.write("[\n" if pretty else "[")
        first = True

        for i in range(count):
            ts                             = rand_timestamp(cfg.start_year, cfg.end_year)
            convo, injected, asst_count    = make_conversation(ts, cfg)
            total_injected                += injected
            total_assistant               += asst_count

            if not first:
                f.write(",\n" if pretty else ",")
            else:
                first = False

            blob = json.dumps(convo, ensure_ascii=False, indent=indent, separators=separators)
            # Indent each line by 2 spaces so objects sit correctly inside the array
            if pretty:
                blob = "  " + blob.replace("\n", "\n  ")
            f.write(blob)

            if report_every and (i + 1) % report_every == 0:
                _progress(i, count, t0)

        f.write("\n]\n" if pretty else "]\n")

    print()
    return total_injected, total_assistant


def write_ndjson_stream(
    out_path:     Path,
    count:        int,
    report_every: int,
    cfg:          ConvConfig,
) -> Tuple[int, int]:
    """Write one conversation JSON object per line.
    Returns (total_injected, total_assistant)."""
    total_injected  = 0
    total_assistant = 0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(count):
            ts                          = rand_timestamp(cfg.start_year, cfg.end_year)
            convo, injected, asst_count = make_conversation(ts, cfg)
            total_injected             += injected
            total_assistant            += asst_count
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")

            if report_every and (i + 1) % report_every == 0:
                _progress(i, count, t0)

    print()
    return total_injected, total_assistant

    print()
    return total_injected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic OpenAI conversations.json export."
    )
    parser.add_argument("--version",        action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--count",    "-n", type=int, default=1000,                        help="Number of conversations (default: 1000)")
    parser.add_argument("--output",   "-o", type=str, default="conversations.json",        help="Output file path")
    parser.add_argument("--seed",     "-s", type=int, default=None,                        help="Random seed for reproducible output")
    parser.add_argument("--start-year",     type=int, default=2023,                        help="Earliest year for timestamps (default: 2023)")
    parser.add_argument("--end-year",       type=int, default=2025,                        help="Latest year for timestamps (default: 2025)")
    parser.add_argument("--min-turns",      type=int, default=1,                           help="Min user/assistant pairs per conversation (default: 1)")
    parser.add_argument("--max-turns",      type=int, default=6,                           help="Max user/assistant pairs per conversation (default: 6)")
    parser.add_argument("--pretty",         action="store_true",                           help="Pretty-print JSON array (default)")
    parser.add_argument("--compact",        action="store_true",                           help="Minify JSON (smallest size)")
    parser.add_argument("--ndjson",         action="store_true",                           help="Write NDJSON (one conversation per line)")
    parser.add_argument("--progress-every", type=int, default=0,                           help="Progress interval (default: auto)")
    parser.add_argument("--inject-code",    action="store_true",                           help="Inject code blocks into assistant replies")
    parser.add_argument("--code-percent",   type=int, default=25,                          help="Percent of assistant replies that get code (default: 25)")
    parser.add_argument("--code-types",     type=str, default="terraform,hcl,yaml,bash,python",  help="Comma-separated languages: terraform,hcl,yaml,bash,python (default: all)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    count    = args.count
    out_path = Path(args.output)
    pretty   = args.pretty or (not args.compact and not args.ndjson)

    report_every = (
        args.progress_every if args.progress_every > 0
        else (max(100, count // 20) if count >= 2000 else max(1, count // 20))
    )

    cfg = ConvConfig(
        start_year   = args.start_year,
        end_year     = args.end_year,
        min_turns    = args.min_turns,
        max_turns    = args.max_turns,
        inject_code  = args.inject_code,
        code_percent = max(0, min(100, args.code_percent)),
        code_types   = _parse_code_types(args.code_types),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"generate_conversations.py v{VERSION}")
    print(f"Generating  : {count:,} conversations")
    print(f"Output      : {out_path}")
    print(f"Timestamps  : {cfg.start_year} .. {cfg.end_year}")
    print(f"Turns       : {cfg.min_turns} .. {cfg.max_turns}")
    print(f"Mode        : {'NDJSON' if args.ndjson else ('compact' if args.compact else 'pretty')}")
    print(f"Inject code : {'on  ('+str(cfg.code_percent)+'%  ['+', '.join(cfg.code_types)+'])' if cfg.inject_code else 'off'}")
    print()

    t0 = time.time()

    if args.ndjson:
        total_injected, total_assistant = write_ndjson_stream(out_path, count, report_every, cfg)
    else:
        total_injected, total_assistant = write_json_array_stream(out_path, count, pretty, args.compact, report_every, cfg)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1_048_576.0

    print(f"Done.")
    print(f"  Conversations  : {count:,}")
    print(f"  File size      : {size_mb:.1f} MB")
    print(f"  Elapsed        : {elapsed:.1f}s")
    if cfg.inject_code:
        pct = total_injected / total_assistant * 100 if total_assistant else 0
        print(f"  Assistant msgs : {total_assistant:,}")
        print(f"  Code injected  : {total_injected:,}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()