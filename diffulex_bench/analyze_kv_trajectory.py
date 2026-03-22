#!/usr/bin/env python3
"""
Analyze ``trajectory.json`` from diffulex_bench (with ``save_kv_mapping_trace``),
typically under ``<output_dir>/run_<timestamp>_<task>/trajectory.json``.

Detects common silent bugs: tensor length mismatch, duplicate physical slots within one
forward, invalid page ids, padding anomalies, and DLLM block span mismatches.

Cross-request slot aliasing in the same GPU step is not represented in this file
(one trajectory per prompt); use engine-side multi-req dumps for that.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator


@dataclass
class Issue:
    kind: str
    message: str
    batch_idx: int = -1
    prompt_idx: int = -1
    step_id: int | None = None
    extra: dict = field(default_factory=dict)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_trajectories(data: Any) -> Generator[tuple[int, int, dict], None, None]:
    """Yield (batch_idx, prompt_idx, trajectory_dict)."""
    if isinstance(data, list):
        for i, tr in enumerate(data):
            yield -1, i, tr
        return
    if isinstance(data, dict):
        if "batches" in data:
            for bi, batch in enumerate(data["batches"]):
                trs = batch.get("trajectories") or []
                for pi, tr in enumerate(trs):
                    yield bi, pi, tr
            return
        if "trajectories" in data:
            for pi, tr in enumerate(data["trajectories"]):
                yield -1, pi, tr
            return
    raise ValueError("Unrecognized trajectory.json layout (expected list or {batches:[...]}).")


def _suffix_all_neg_one(xs: list[int]) -> bool:
    if not xs:
        return True
    seen_neg = False
    for x in xs:
        if x < 0:
            seen_neg = True
        elif seen_neg:
            return False
    return True


def analyze_step_kv(
    batch_idx: int,
    prompt_idx: int,
    step: dict,
    page_size: int,
) -> Iterator[Issue]:
    step_id = step.get("step_id")
    kv = step.get("kv_mapping_trace")
    if kv is None:
        return
    sm = kv.get("slot_mapping")
    tp = kv.get("token_positions")
    ft = kv.get("forward_token_ids")
    if not isinstance(sm, list) or not isinstance(tp, list) or not isinstance(ft, list):
        yield Issue(
            "bad_types",
            "kv_mapping_trace slot_mapping/token_positions/forward_token_ids must be lists",
            batch_idx,
            prompt_idx,
            step_id,
        )
        return
    if not (len(sm) == len(tp) == len(ft)):
        yield Issue(
            "length_mismatch",
            f"len(slot_mapping)={len(sm)} len(token_positions)={len(tp)} len(forward_token_ids)={len(ft)}",
            batch_idx,
            prompt_idx,
            step_id,
        )

    pos_slots = [s for s in sm if isinstance(s, int) and s >= 0]
    if pos_slots:
        ctr = Counter(pos_slots)
        dups = sorted([s for s, c in ctr.items() if c > 1])
        if dups:
            yield Issue(
                "duplicate_slot_same_forward",
                "Same physical slot index appears more than once in one forward (likely KV corruption risk)",
                batch_idx,
                prompt_idx,
                step_id,
                {"duplicate_slots": dups[:32], "num_duplicates": len(dups)},
            )

    if sm and not _suffix_all_neg_one(sm):
        yield Issue(
            "slot_padding_not_suffix",
            "Negative slot_mapping (-1) is not confined to a trailing suffix",
            batch_idx,
            prompt_idx,
            step_id,
        )

    pt = kv.get("page_table")
    if isinstance(pt, list) and page_size > 0 and pos_slots:
        pages = {int(p) for p in pt if isinstance(p, int) and p >= 0}
        bad = sorted({s // page_size for s in pos_slots if (s // page_size) not in pages})
        if bad:
            yield Issue(
                "slot_page_not_in_page_table",
                "slot // page_size not found in page_table for some active slots",
                batch_idx,
                prompt_idx,
                step_id,
                {"missing_derived_pages": bad[:32]},
            )

    db = kv.get("dllm_blocks_trace")
    if isinstance(db, dict):
        phase = db.get("phase")
        if phase == "prefill":
            blocks = (db.get("all_blocks") or []) if isinstance(db.get("all_blocks"), list) else []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                tids = b.get("token_ids")
                start, end = b.get("start"), b.get("end")
                if isinstance(tids, list) and isinstance(start, int) and isinstance(end, int):
                    if len(tids) != end - start:
                        yield Issue(
                            "dllm_block_token_len",
                            f"block_id={b.get('block_id')}: len(token_ids)={len(tids)} != end-start={end - start}",
                            batch_idx,
                            prompt_idx,
                            step_id,
                        )
        elif phase == "decode":
            buf = db.get("buffer")
            if isinstance(buf, dict):
                blks = buf.get("blocks") or []
                if isinstance(blks, list):
                    for b in blks:
                        if not isinstance(b, dict):
                            continue
                        tids = b.get("token_ids")
                        start, end = b.get("start"), b.get("end")
                        if isinstance(tids, list) and isinstance(start, int) and isinstance(end, int):
                            if len(tids) != end - start:
                                yield Issue(
                                    "dllm_buffer_block_token_len",
                                    f"block_id={b.get('block_id')}: len(token_ids)={len(tids)} != end-start={end - start}",
                                    batch_idx,
                                    prompt_idx,
                                    step_id,
                                )


def analyze_file(path: Path) -> list[Issue]:
    data = load_json(path)
    issues: list[Issue] = []
    steps_with_kv = 0
    steps_total = 0

    for batch_idx, prompt_idx, traj in iter_trajectories(data):
        steps = traj.get("trajectory") or []
        if not isinstance(steps, list):
            issues.append(
                Issue("bad_trajectory", "trajectory is not a list", batch_idx, prompt_idx, None)
            )
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            steps_total += 1
            if step.get("kv_mapping_trace"):
                steps_with_kv += 1
            ps = 32
            kv = step.get("kv_mapping_trace")
            if isinstance(kv, dict) and isinstance(kv.get("page_size"), int):
                ps = kv["page_size"]
            issues.extend(analyze_step_kv(batch_idx, prompt_idx, step, ps))

    if steps_total > 0 and steps_with_kv == 0:
        issues.insert(
            0,
            Issue(
                "no_kv_trace",
                "No step contained kv_mapping_trace. Enable Config.save_kv_mapping_trace and multi_bd decoding.",
                -1,
                -1,
                None,
            ),
        )

    return issues


def format_issue(i: Issue) -> str:
    loc = f"batch={i.batch_idx} prompt={i.prompt_idx}"
    if i.step_id is not None:
        loc += f" step={i.step_id}"
    extra = f" {i.extra}" if i.extra else ""
    return f"[{i.kind}] {loc}: {i.message}{extra}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-json", type=Path, help="Path to trajectory.json")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write issues as JSON array to this path",
    )
    args = p.parse_args(argv)

    if not args.trajectory_json.is_file():
        print(f"File not found: {args.trajectory_json}", file=sys.stderr)
        return 2

    issues = analyze_file(args.trajectory_json)
    if args.json_out:
        args.json_out.write_text(
            json.dumps([i.__dict__ for i in issues], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if not issues:
        print("No issues found (within the checks implemented in this script).")
        return 0

    for i in issues:
        print(format_issue(i))
    # Non-zero exit if anything beyond informational no_kv_trace when steps exist
    serious = [i for i in issues if i.kind != "no_kv_trace"]
    return 1 if serious else 0


if __name__ == "__main__":
    raise SystemExit(main())
