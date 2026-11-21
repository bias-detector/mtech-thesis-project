"""
Drop vandalism / revert cycles from edit pairs.

How it works (3 steps):

  1) Chain edits by NORMALIZED text (ignore spaces), oldest -> newest.
     So "wankers ," and "wankers," still connect as the same tip.

  2) If earlier text reappears within REVERT_WINDOW steps
     (A->B->A or A->B->C->A), discard EVERY edge in that cycle.

  3) Poison whole revision bursts by diff id (parsed from diff_url):
     - If revision X is in a discarded cycle, drop ALL pairs with that
       same diff id (sibling vandal lines never restored as exact text).
     - Also drop bridge revisions between vandalism and restore
       (vandal_diff -> bridge -> restore_diff), e.g. egory->eat.

Reads:  data/03_changed/*.json
Writes:
  survived  -> data/04_vandal_free/<article>.json
  discarded -> data/discarded/04_vandalised/<article>.json  (only if any dropped)

Pair order in each file: bottom = edited first, top = edited last.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any

from pipeline.utils.diff_url import pair_diff_id, pair_old_id, revision_time_key
from pipeline.utils.io import create_output_folder, save_article, article_envelope, resolve_article_files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "03_changed")
SURVIVED_FOLDER = os.path.join(BASE_DIR, "data", "04_vandal_free")
DISCARDED_FOLDER = os.path.join(BASE_DIR, "data", "discarded", "04_vandalised")

REVERT_WINDOW = 5


def revert_key(text: Any) -> str:
    """Strip all whitespace + lowercase so spacing-only differences match."""
    return re.sub(r"\s+", "", str(text or "")).lower()


def pick_chronological_successor(
    candidate_indices: list[int],
    pairs: list[dict[str, Any]],
    current_index: int,
) -> int | None:
    """Among unused edges with the same tip, pick the next later edit."""
    current_key = revision_time_key(pairs[current_index])
    later = [
        j
        for j in candidate_indices
        if revision_time_key(pairs[j]) > current_key
    ]
    if not later:
        return None
    return min(later, key=lambda j: revision_time_key(pairs[j]))


def _collapse_cycle_edges(
    versions: list[str],
    keys: list[str],
    edge_indices: list[int],
    revert_window: int,
) -> tuple[list[str], list[str], list[int], set[int], list[tuple[int, int]]]:
    """
    If a normalized version reappears within revert_window steps, discard
    every edge in that cycle and truncate the chain. Repeat until stable.

    Also returns cycle_spans: (first_edge_index, last_edge_index) for each
    collapsed cycle so we know vandalism vs restore endpoints.
    """
    discarded: set[int] = set()
    cycle_spans: list[tuple[int, int]] = []
    changed = True
    while changed:
        changed = False
        n = len(versions)
        for i in range(n):
            upper = min(i + revert_window + 1, n)
            for j in range(i + 1, upper):
                if keys[i] != keys[j]:
                    continue
                # edge_indices[k] is the pair versions[k] -> versions[k+1]
                span_edges = edge_indices[i:j]
                for eidx in span_edges:
                    discarded.add(eidx)
                if span_edges:
                    cycle_spans.append((span_edges[0], span_edges[-1]))
                del versions[i + 1 : j + 1]
                del keys[i + 1 : j + 1]
                del edge_indices[i:j]
                changed = True
                break
            if changed:
                break
    return versions, keys, edge_indices, discarded, cycle_spans


def _find_cycle_discard_indices(
    pairs: list[dict[str, Any]],
    revert_window: int,
) -> tuple[set[int], set[str], set[str]]:
    """
    Step 1+2: chain on normalized tips, discard restore-cycle edges.

    Returns:
      discarded indices,
      vandal_diff_ids  (first edge of each cycle — the corrupt edit),
      restore_diff_ids (last edge of each cycle — the undo edit).
    """
    by_base_key: dict[str, list[int]] = defaultdict(list)
    base_keys = [revert_key(p.get("base")) for p in pairs]
    revised_keys = [revert_key(p.get("revised")) for p in pairs]
    for index, key in enumerate(base_keys):
        by_base_key[key].append(index)

    used: set[int] = set()
    discarded: set[int] = set()
    vandal_diff_ids: set[str] = set()
    restore_diff_ids: set[str] = set()

    start_order = sorted(
        range(len(pairs)), key=lambda i: revision_time_key(pairs[i])
    )

    def _remember_span(first_idx: int, last_idx: int) -> None:
        vd = pair_diff_id(pairs[first_idx])
        rd = pair_diff_id(pairs[last_idx])
        if vd is not None:
            vandal_diff_ids.add(vd)
        if rd is not None:
            restore_diff_ids.add(rd)

    for start in start_order:
        if start in used or start in discarded:
            continue

        versions = [
            str(pairs[start].get("base", "")),
            str(pairs[start].get("revised", "")),
        ]
        keys = [base_keys[start], revised_keys[start]]
        edge_indices = [start]
        used.add(start)
        current_index = start

        while True:
            tip_key = keys[-1]
            candidates = [
                j for j in by_base_key.get(tip_key, []) if j not in used
            ]
            if not candidates:
                break

            nxt_index = pick_chronological_successor(
                candidates, pairs, current_index
            )
            if nxt_index is None:
                break

            rk = revised_keys[nxt_index]
            revised_display = str(pairs[nxt_index].get("revised", ""))
            if revised_display in versions:
                n = len(keys)
                lower = max(0, n - revert_window)
                if not any(keys[i] == rk for i in range(lower, n)):
                    break

            used.add(nxt_index)
            versions.append(revised_display)
            keys.append(rk)
            edge_indices.append(nxt_index)
            current_index = nxt_index

            (
                versions,
                keys,
                edge_indices,
                cycle_drop,
                spans,
            ) = _collapse_cycle_edges(
                versions, keys, edge_indices, revert_window
            )
            discarded.update(cycle_drop)
            for first_idx, last_idx in spans:
                _remember_span(first_idx, last_idx)
            if cycle_drop:
                if not edge_indices:
                    break
                current_index = edge_indices[-1]

    # Safety net: remaining mutual reverse A↔B (normalized) — drop both.
    by_direction: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(pairs)):
        if i in discarded:
            continue
        a, b = base_keys[i], revised_keys[i]
        if not a and not b:
            continue
        by_direction[(a, b)].append(i)

    seen_undirected: set[tuple[str, str]] = set()
    for (a, b), idxs in by_direction.items():
        if a == b:
            continue
        rev = (b, a)
        if rev not in by_direction:
            continue
        undirected = (a, b) if a < b else (b, a)
        if undirected in seen_undirected:
            continue
        seen_undirected.add(undirected)

        # Older direction = vandalism, newer = restore
        ab_first = min(idxs, key=lambda i: revision_time_key(pairs[i]))
        ba_first = min(
            by_direction[rev], key=lambda i: revision_time_key(pairs[i])
        )
        if revision_time_key(pairs[ab_first]) <= revision_time_key(
            pairs[ba_first]
        ):
            vandal_i, restore_i = ab_first, ba_first
        else:
            vandal_i, restore_i = ba_first, ab_first
        _remember_span(vandal_i, restore_i)

        discarded.update(idxs)
        discarded.update(by_direction[rev])

    return discarded, vandal_diff_ids, restore_diff_ids


def _poison_revision_bursts(
    pairs: list[dict[str, Any]],
    cycle_discarded: set[int],
    vandal_diff_ids: set[str],
    restore_diff_ids: set[str],
) -> set[int]:
    """
    Step 3: expand discarded cycles to whole revision bursts.

    a) Same-diff burst: if revision X is in a cycle, drop all pairs
       that share that diff id (sibling lines in the same edit).

    b) Bridge revisions between vandalism and restore:
         vandal_diff -> bridge_diff -> restore_diff
       Drop the bridge too (e.g. egory:Hacks -> eat my asshole).
       Bridges start from a vandalism id, never from a restore id, so
       later legitimate edits after a restore are not wiped.
    """
    bad_diff_ids: set[str] = set(vandal_diff_ids) | set(restore_diff_ids)
    for i in cycle_discarded:
        did = pair_diff_id(pairs[i])
        if did is not None:
            bad_diff_ids.add(did)

    by_old: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        oid = pair_old_id(pair)
        if oid is not None:
            by_old[oid].append(i)

    # Seeds that may start a bridge: vandalism revisions (and later bridges).
    bridge_parents = set(vandal_diff_ids)

    changed = True
    while changed:
        changed = False
        for pair in pairs:
            old_s = pair_old_id(pair) or ""
            diff_s = pair_diff_id(pair) or ""
            if not old_s or not diff_s or diff_s in bad_diff_ids:
                continue
            # Must start from vandalism/bridge — never from a pure restore.
            if old_s not in bridge_parents:
                continue

            for j in by_old.get(diff_s, []):
                child_diff = pair_diff_id(pairs[j]) or ""
                if child_diff in restore_diff_ids:
                    bad_diff_ids.add(diff_s)
                    bridge_parents.add(diff_s)
                    changed = True
                    break

    discarded = set(cycle_discarded)
    for i, pair in enumerate(pairs):
        did = pair_diff_id(pair)
        if did is not None and did in bad_diff_ids:
            discarded.add(i)
    return discarded


def split_vandalism_pairs(
    pairs: list[dict[str, Any]],
    revert_window: int = REVERT_WINDOW,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kept, discarded) after cycle detection + revision-burst poison."""
    if not pairs:
        return [], []

    cycle_discarded, vandal_ids, restore_ids = _find_cycle_discard_indices(
        pairs, revert_window
    )
    discarded_indices = _poison_revision_bursts(
        pairs, cycle_discarded, vandal_ids, restore_ids
    )

    kept = [p for i, p in enumerate(pairs) if i not in discarded_indices]
    discarded = [p for i, p in enumerate(pairs) if i in discarded_indices]
    return kept, discarded


def process_folder(
    input_folder: str,
    survived_folder: str,
    discarded_folder: str,
    article: str | None = None,
) -> None:
    files = resolve_article_files(input_folder, article)
    if article and not files:
        print(f"Article not found in {input_folder}: {article}")
        return

    written = 0
    skipped = 0
    pairs_in = 0
    pairs_kept = 0
    pairs_discarded = 0

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                article = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Skipping {file_name}: could not read JSON")
            skipped += 1
            continue

        raw_pairs = list(article.get("pairs") or [])
        kept, discarded = split_vandalism_pairs(raw_pairs)
        pairs_in += len(raw_pairs)
        pairs_kept += len(kept)
        pairs_discarded += len(discarded)

        if not kept:
            skipped += 1
            if discarded and save_article(
                article_envelope(article, discarded), discarded_folder
            ):
                pass
            continue

        if not save_article(article_envelope(article, kept), survived_folder):
            skipped += 1
            continue

        written += 1
        if discarded:
            save_article(article_envelope(article, discarded), discarded_folder)

        if written % 100 == 0:
            print(f"Wrote {written} articles...")

    print(f"Done. Survived articles: {written} -> {survived_folder}")
    print(f"Skipped (no survived pairs / bad JSON): {skipped}")
    print(
        f"Pairs: {pairs_in} -> kept {pairs_kept}, "
        f"discarded {pairs_discarded} -> {discarded_folder}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop vandalism / revert cycle pairs."
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help='Process one article only, e.g. "Hack" or "Hack.json"',
    )
    args = parser.parse_args()

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    create_output_folder(SURVIVED_FOLDER)
    create_output_folder(DISCARDED_FOLDER)
    process_folder(
        INPUT_FOLDER, SURVIVED_FOLDER, DISCARDED_FOLDER, article=args.article
    )


if __name__ == "__main__":
    main()
