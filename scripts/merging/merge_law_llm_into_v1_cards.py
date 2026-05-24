#!/usr/bin/env python
"""Merge LLM descriptors into v1 static law cards → unified v2 corpus.

Counterpart of `merge_llm_enrichment_into_v4_cards.py` but for laws.

For every static card:
  - If its `_source_row` is in the LLM index → emit `enrichment_source="llm+static"`.
  - Otherwise                                 → emit `enrichment_source="static"`.

Boilerplate suppression: when the merged provision role resolves to
`transitional_or_commencement`, `fees_or_costs`, or `data_reporting`, or when
the source text is essentially "Aufgehoben" / "Tritt … in Kraft", the rule
fields (legal_rule, applicability_conditions, exceptions_or_limitations,
legal_question) are forcibly blanked even if the LLM produced something.

Usage:
    python scripts/merge_law_llm_into_v1_cards.py \\
        --llm artifacts/law_llm_descriptors.jsonl

Smoke:
    python scripts/merge_law_llm_into_v1_cards.py --llm <path> --limit 1000 \\
        --output artifacts/law_authority_cards_v2_unified.smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

DEFAULT_CARDS = ART_DIR / "law_authority_cards_v1_static.jsonl"
DEFAULT_OUTPUT = ART_DIR / "law_authority_cards_v2_unified.jsonl"
DEFAULT_SUMMARY = ART_DIR / "law_authority_cards_v2_unified.summary.json"

BOILERPLATE_ROLES = {"transitional_or_commencement", "fees_or_costs", "data_reporting"}
RULE_FIELDS = ["legal_rule", "applicability_conditions",
               "exceptions_or_limitations", "legal_question"]

REPEAL_RE = re.compile(r"\b(aufgehoben|abrog[ée]|abrogato)\b", re.IGNORECASE)
COMMENCEMENT_RE = re.compile(r"tritt\s+(?:am|per)\s+\d", re.IGNORECASE)


def load_llm_index(path: Path) -> dict[int, dict]:
    print(f"[load] reading LLM descriptors from {path}", flush=True)
    index: dict[int, dict] = {}
    counts = Counter()
    t0 = time.time()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counts["bad_json"] += 1
                continue
            src = rec.get("_source_row")
            if src is None:
                counts["missing_source_row"] += 1
                continue
            status = (rec.get("llm_generation") or {}).get("status", "")
            if status.startswith("failed"):
                counts["failed_status"] += 1
                continue
            llm = rec.get("llm_enrichment") or {}
            if not isinstance(llm, dict) or not llm:
                counts["empty_enrichment"] += 1
                continue
            index[int(src)] = rec
            counts["loaded"] += 1
    print(f"[load] loaded={counts['loaded']:,}  failed={counts['failed_status']:,}  "
          f"empty={counts['empty_enrichment']:,}  bad_json={counts['bad_json']:,}  "
          f"in {time.time()-t0:.1f}s", flush=True)
    return index


def is_boilerplate(card: dict, llm_role: str) -> bool:
    if llm_role in BOILERPLATE_ROLES:
        return True
    static_roles = set((card.get("rag_enrichment") or {}).get("provision_roles_static")
                       or card.get("provision_roles") or [])
    if static_roles and static_roles.issubset(BOILERPLATE_ROLES):
        return True
    text = (card.get("retrieval_views") or {}).get("raw_context") or ""
    if not text.strip():
        return True
    if REPEAL_RE.search(text) and len(text) < 200:
        return True
    if COMMENCEMENT_RE.search(text) and len(text) < 200:
        return True
    return False


def build_views(card: dict, merged_rag: dict) -> dict:
    """Augment static retrieval_views with LLM-derived views."""
    views = dict(card.get("retrieval_views") or {})
    es = merged_rag.get("english_summary", "")
    if es:
        views["english_summary_view"] = es
    rule = merged_rag.get("legal_rule", "")
    if rule:
        views["legal_rule_view"] = rule
    cond = merged_rag.get("applicability_conditions") or []
    excs = merged_rag.get("exceptions_or_limitations") or []
    if cond or excs:
        parts = []
        if cond:
            parts.append("applies: " + " | ".join(cond))
        if excs:
            parts.append("exceptions: " + " | ".join(excs))
        views["rule_components_view"] = " || ".join(parts)
    concepts = merged_rag.get("concepts_en") or []
    if concepts:
        views["concepts_en_view"] = " | ".join(concepts)
    terms = merged_rag.get("terms_de_to_en") or []
    if terms:
        views["terms_bilingual_view"] = " | ".join(
            f"{t.get('de','')}→{t.get('en','')}" for t in terms if t.get("de") and t.get("en")
        )
    addressees = merged_rag.get("addressees") or []
    if addressees:
        views["addressees_view"] = " | ".join(addressees)
    return views


def merge(args: argparse.Namespace) -> dict:
    llm_index = load_llm_index(args.llm)

    stats = Counter()
    source_dist = Counter()
    role_dist = Counter()
    suppression_dist = Counter()
    field_fill = Counter()
    field_fill_by_source = {"static": Counter(), "llm+static": Counter()}

    t0 = time.time()
    print(f"[merge] streaming {args.cards} → {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.cards.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["scanned"] += 1
            if args.limit and stats["scanned"] > args.limit:
                break
            try:
                card = json.loads(line)
            except Exception:
                stats["card_parse_error"] += 1
                continue

            src = int(card.get("_source_row", -1))
            llm_rec = llm_index.get(src)
            llm = (llm_rec or {}).get("llm_enrichment") or {}
            llm_quality = (llm_rec or {}).get("llm_quality") or {}
            enrichment_source = "llm+static" if llm else "static"
            source_dist[enrichment_source] += 1

            static_rag = card.get("rag_enrichment") or {}

            # Build merged rag_enrichment: static fields take precedence on the
            # static keys; LLM fields are layered on top.
            merged_rag = dict(static_rag)
            for k in [
                "english_summary", "legal_rule", "applicability_conditions",
                "exceptions_or_limitations", "legal_question", "concepts_en",
                "terms_de_to_en", "defined_terms", "addressees",
                "sanctions_or_consequences", "provision_role_llm",
                "specificity_score",
            ]:
                if k in llm:
                    merged_rag[k] = llm[k]

            llm_role = llm.get("provision_role_llm", "")
            boiler = is_boilerplate(card, llm_role)
            if boiler:
                for k in RULE_FIELDS:
                    if isinstance(merged_rag.get(k), list):
                        merged_rag[k] = []
                    else:
                        merged_rag[k] = ""
                suppression_dist["rule_fields_suppressed"] += 1

            # Build augmented retrieval views
            views = build_views(card, merged_rag)

            # Quality block
            quality = dict(card.get("enrichment_quality") or {})
            quality.update({
                "llm_status": (llm_rec or {}).get("llm_generation", {}).get("status", "skipped"),
                "llm_json_valid": llm_quality.get("json_valid", False) if llm else False,
                "terms_grounded_pct": llm_quality.get("terms_grounded_pct", 0.0) if llm else 0.0,
                "boilerplate_role": boiler,
                "rule_fields_suppressed": boiler,
            })

            out_rec = {
                "_source_row": src,
                "citation": card.get("citation", ""),
                "language": card.get("language", "de"),
                "family": card.get("family", "law"),
                "pattern": card.get("pattern", ""),
                "law_title": card.get("law_title", ""),
                "title_section_path": card.get("title_section_path", ""),
                "structural": card.get("structural", {}),
                "title_metadata": card.get("title_metadata", {}),
                "enrichment_source": enrichment_source,
                "rag_enrichment": merged_rag,
                "normalized_anchors": card.get("normalized_anchors", {}),
                "retrieval_views": views,
                "enrichment_quality": quality,
                "provenance": card.get("provenance", {}),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            stats["written"] += 1

            role = (merged_rag.get("provision_role_llm")
                    or (merged_rag.get("provision_roles_static") or [""])[0]
                    or "other")
            role_dist[role] += 1

            for k, v in merged_rag.items():
                non_empty = bool(v) if not isinstance(v, (int, float)) else (v != 0)
                if non_empty:
                    field_fill[k] += 1
                    field_fill_by_source[enrichment_source][k] += 1

            if stats["scanned"] % args.progress_every == 0:
                rate = stats["scanned"] / max(time.time() - t0, 1e-9)
                print(f"  scanned={stats['scanned']:>8,}  written={stats['written']:>8,}  "
                      f"({rate:>5.0f} rec/s)", flush=True)

    elapsed = time.time() - t0
    summary = {
        "inputs": {"cards": str(args.cards), "llm": str(args.llm)},
        "output": str(args.output),
        "elapsed_seconds": round(elapsed, 1),
        "stats": dict(stats),
        "enrichment_source_counts": dict(source_dist),
        "provision_role_distribution": role_dist.most_common(),
        "suppression": dict(suppression_dist),
        "rag_field_fill_total": field_fill.most_common(),
        "rag_field_fill_pct_by_source": {
            src: {k: round(c / max(source_dist.get(src, 1), 1) * 100, 2) for k, c in counter.items()}
            for src, counter in field_fill_by_source.items()
        },
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] scanned={stats['scanned']:,}  written={stats['written']:,}  "
          f"in {elapsed/60:.1f} min", flush=True)
    print(f"[done] static-only: {source_dist.get('static',0):,}", flush=True)
    print(f"[done] llm+static : {source_dist.get('llm+static',0):,}", flush=True)
    print(f"[done] output:      {args.output}", flush=True)
    print(f"[done] summary:     {args.summary}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", type=Path, required=True,
                    help="Path to law_llm_descriptors.jsonl")
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--progress-every", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.cards.exists():
        raise SystemExit(f"static cards not found: {args.cards}")
    if not args.llm.exists():
        raise SystemExit(f"LLM descriptors not found: {args.llm}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
