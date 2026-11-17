from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def canonical_roster(values: Iterable[object]) -> list[str]:
    # Preserve identifier bytes as strings: the preprocessing pipeline uses
    # astype(str) without trimming, so trimming here could change an ID.
    roster = [str(value) for value in values]
    if any(value.strip() == "" for value in roster):
        raise ValueError("The public client roster contains an empty identifier.")
    if len(roster) != len(set(roster)):
        raise ValueError("The public client roster contains duplicate identifiers.")
    return sorted(roster)


def hash_bucket(client_id: str, seed: int, n_virtual_clients: int) -> int:
    """Map a public source-client ID without consulting transaction-level data."""
    if n_virtual_clients < 1:
        raise ValueError("n_virtual_clients must be >= 1")
    payload = f"TA-DPFL-virtual-client-v1\x1f{int(seed)}\x1f{client_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"TA-DPFL-VC-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % n_virtual_clients


def build_assignments(roster: Iterable[object], seed: int, n_virtual_clients: int) -> list[dict[str, str]]:
    clients = canonical_roster(roster)
    return [
        {
            "source_client_id": client_id,
            "virtual_client_id": f"VC{hash_bucket(client_id, seed, n_virtual_clients):04d}",
        }
        for client_id in clients
    ]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_roster(config: dict, roster_file: Path | None) -> list[str]:
    configured = config.get("data", {}).get("public_client_ids")
    if configured:
        return canonical_roster(configured)
    if roster_file is not None:
        values = json.loads(roster_file.read_text(encoding="utf-8-sig"))
        if not isinstance(values, list):
            raise ValueError("--roster-file must contain a JSON array of client IDs")
        return canonical_roster(values)
    fallback = Path("outputs/public_client_ids_candidate.json")
    if fallback.exists():
        values = json.loads(fallback.read_text(encoding="utf-8-sig"))
        if not isinstance(values, list):
            raise ValueError(f"{fallback} must contain a JSON array")
        return canonical_roster(values)
    raise ValueError(
        "No public client roster found. Use a resolved config whose data.public_client_ids "
        "contains the complete frozen roster, or pass --roster-file."
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_client_id", "virtual_client_id"])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic, data-independent source-client to virtual-client map."
    )
    parser.add_argument("--config", default="configs/experiment.resolved.yaml")
    parser.add_argument("--roster-file", default=None, help="Optional JSON array; used only if config roster is absent.")
    parser.add_argument("--num-virtual-clients", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to top-level config seed.")
    parser.add_argument("--output-dir", default="outputs/virtual_federation")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Configuration YAML must contain a mapping")

    roster_file = Path(args.roster_file) if args.roster_file else None
    roster = load_roster(config, roster_file)
    seed = int(config.get("seed", 2026) if args.seed is None else args.seed)
    if args.num_virtual_clients < 1:
        raise ValueError("--num-virtual-clients must be >= 1")
    if args.num_virtual_clients > len(roster):
        raise ValueError(
            f"Requested {args.num_virtual_clients} virtual clients for only {len(roster)} source clients."
        )

    assignments = build_assignments(roster, seed, args.num_virtual_clients)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "virtual_client_manifest.csv"
    metadata_path = output_dir / "virtual_client_manifest_metadata.json"
    write_csv(manifest_path, assignments)

    roster_text = json.dumps(roster, ensure_ascii=False, separators=(",", ":"))
    manifest_text = json.dumps(assignments, ensure_ascii=False, separators=(",", ":"))
    metadata = {
        "schema_version": 1,
        "assignment_method": "BLAKE2b-64 hash bucket of public source_client_id and fixed seed",
        "hash_person": "TA-DPFL-VC-v1",
        "seed": seed,
        "num_virtual_clients_requested": args.num_virtual_clients,
        "num_source_clients": len(roster),
        "virtual_client_id_format": "VC0000-based, zero-padded to four digits",
        "roster_sha256": sha256_text(roster_text),
        "assignment_sha256": sha256_text(manifest_text),
        "assignment_uses_transaction_rows": False,
        "assignment_uses_labels_or_typologies": False,
        "assignment_uses_per_client_transaction_counts": False,
        "privacy_note": (
            "Mapping is deterministic from the fixed public roster and seed only. "
            "It does not balance using private transaction counts. Audit bucket sizes before training."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    unique_virtual = len({row["virtual_client_id"] for row in assignments})
    print(f"Source clients: {len(roster):,}")
    print(f"Requested virtual clients: {args.num_virtual_clients:,}")
    print(f"Virtual buckets occupied by roster IDs: {unique_virtual:,}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")
    print("Assignment depends only on the public roster IDs and fixed seed.")
    print("No transaction labels, typologies, or per-client row counts were used.")


if __name__ == "__main__":
    main()
