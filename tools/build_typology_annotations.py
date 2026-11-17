#!/usr/bin/env python3
"""Build source_row_id/typology annotations from IBM HI-Small_Patterns.txt.

This joins pattern rows to the original HI-Small_Trans.csv by the complete
11-field transaction row, preserving CSV field text and original row order.
It intentionally fails on ambiguous duplicate transaction rows, conflicting
cross-typology membership, malformed pattern blocks, or unmatched pattern rows.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EXPECTED_COLUMNS = 11
LABEL_INDEX = 10
VALID = {
    "FAN-OUT", "FAN-IN", "CYCLE", "BIPARTITE", "STACK", "RANDOM",
    "SCATTER-GATHER", "GATHER-SCATTER",
}
BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*(?P<typology>[A-Za-z_-]+)(?::\s*(?P<detail>.*))?$", re.I)
END_RE = re.compile(r"^END LAUNDERING ATTEMPT\s*-\s*(?P<typology>[A-Za-z_-]+)\s*$", re.I)


def normalize_typology(raw: str) -> str:
    value = re.sub(r"[_\s]+", "-", raw.strip().upper())
    while "--" in value:
        value = value.replace("--", "-")
    if value not in VALID:
        raise ValueError(f"Unknown typology {raw!r}; expected one of {sorted(VALID)}")
    return value


def row_key(row: Iterable[str]) -> Tuple[str, ...]:
    # csv.reader removes CSV quoting; preserve each decoded field as text.
    # Do not parse/reformat amounts: BTC rows can have six decimal places.
    return tuple(value.strip() for value in row)


def read_patterns(path: Path) -> Dict[Tuple[str, ...], str]:
    labels: Dict[Tuple[str, ...], str] = {}
    active: str | None = None
    saw_begin = 0
    saw_end = 0
    data_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            line = raw_line.strip("\r\n")
            if not line.strip():
                continue
            begin = BEGIN_RE.match(line.strip())
            end = END_RE.match(line.strip())
            if begin:
                if active is not None:
                    raise ValueError(f"Nested BEGIN at {path}:{line_no}; active typology={active}")
                active = normalize_typology(begin.group("typology"))
                saw_begin += 1
                continue
            if end:
                if active is None:
                    raise ValueError(f"END without BEGIN at {path}:{line_no}")
                ending = normalize_typology(end.group("typology"))
                if ending != active:
                    raise ValueError(f"Mismatched END at {path}:{line_no}: {ending} != {active}")
                active = None
                saw_end += 1
                continue
            if active is None:
                raise ValueError(f"Transaction row outside a pattern block at {path}:{line_no}")

            parsed = next(csv.reader([line]))
            if len(parsed) != EXPECTED_COLUMNS:
                raise ValueError(
                    f"Expected {EXPECTED_COLUMNS} transaction fields at {path}:{line_no}; "
                    f"found {len(parsed)}. Check CSV quoting/delimiters."
                )
            if parsed[LABEL_INDEX].strip() not in {"1", "true", "True", "TRUE"}:
                raise ValueError(f"Pattern row is not labelled laundering=1 at {path}:{line_no}")
            key = row_key(parsed)
            previous = labels.get(key)
            if previous is not None and previous != active:
                raise ValueError(
                    "The same exact transaction row occurs under multiple typologies "
                    f"({previous}, {active}) at {path}:{line_no}. The paper's strata are mutually "
                    "exclusive; resolve/document this overlap before training."
                )
            labels[key] = active
            data_rows += 1

    if active is not None:
        raise ValueError(f"Unclosed pattern block at end of {path}: {active}")
    if saw_begin == 0 or saw_begin != saw_end or data_rows == 0:
        raise ValueError(
            f"Invalid/empty pattern file: BEGIN blocks={saw_begin}, END blocks={saw_end}, "
            f"transaction rows={data_rows}"
        )
    print(f"Patterns parsed: {saw_begin} attempts, {data_rows} pattern-row occurrences, "
          f"{len(labels)} unique transaction rows")
    return labels


def build_annotations(trans_path: Path, patterns_path: Path, output_path: Path) -> None:
    labels_by_key = read_patterns(patterns_path)
    matches: Dict[Tuple[str, ...], int] = Counter()
    annotations: List[Tuple[int, str]] = []
    typology_counts: Counter[str] = Counter()

    with trans_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Transaction CSV is empty: {trans_path}")
        if len(header) != EXPECTED_COLUMNS:
            raise ValueError(f"Expected {EXPECTED_COLUMNS} columns in transaction header; found {len(header)}")
        normalized_header = [h.strip().lower() for h in header]
        if "is laundering" not in normalized_header:
            raise ValueError("Could not find 'Is Laundering' in transaction CSV header")
        label_col = normalized_header.index("is laundering")

        for source_row_id, row in enumerate(reader):
            if len(row) != EXPECTED_COLUMNS:
                raise ValueError(
                    f"Transaction CSV row {source_row_id + 2} has {len(row)} fields; "
                    f"expected {EXPECTED_COLUMNS}. The source CSV must remain unchanged."
                )
            key = row_key(row)
            typology = labels_by_key.get(key)
            if typology is None:
                continue
            if row[label_col].strip() not in {"1", "true", "True", "TRUE"}:
                raise ValueError(
                    f"Pattern row matched source_row_id={source_row_id}, but source label is not 1"
                )
            matches[key] += 1
            if matches[key] > 1:
                raise ValueError(
                    "An exact pattern transaction row matched multiple rows in HI-Small_Trans.csv. "
                    f"Ambiguous duplicate source rows for typology={typology}; source_row_id={source_row_id}. "
                    "Resolve this explicitly rather than assigning labels arbitrarily."
                )
            annotations.append((source_row_id, typology))
            typology_counts[typology] += 1

    unmatched = [key for key in labels_by_key if matches.get(key, 0) == 0]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} unique pattern transaction rows did not match the transaction CSV. "
            "The two files may be from different dataset versions, or source formatting/columns differ. "
            f"First unmatched key: {unmatched[0]!r}"
        )
    if not annotations:
        raise ValueError("No typology transactions matched; refusing to create an empty annotation file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_row_id", "typology"])
        writer.writerows(annotations)

    print(f"Annotations written: {len(annotations)} -> {output_path}")
    for typology in sorted(typology_counts):
        print(f"  {typology}: {typology_counts[typology]}")
    print("Unclassified positive transactions are intentionally absent from this file; "
          "the preprocessing pipeline assigns them to UNCL.")
    print("Do not proceed unless these counts reconcile with the dataset release and the paper.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", default="data/HI-Small_Trans.csv", type=Path)
    parser.add_argument("--patterns", default="data/HI-Small_Patterns.txt", type=Path)
    parser.add_argument("--output", default="data/typology_annotations.csv", type=Path)
    args = parser.parse_args()
    try:
        build_annotations(args.transactions, args.patterns, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
