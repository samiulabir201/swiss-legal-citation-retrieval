import json
from pathlib import Path
from collections import Counter

JUDGE = Path(r"E:\swiss_citation_extraction\cache_endgame\judge")

# Pick the worst query (487b - 99% YES) and dump 5 YES responses in full
q = JUDGE / "487b5c29cdf6f10eaffcc3d43087eb383b795331"
yes_full = []
no_full = []
for j in q.iterdir():
    d = json.loads(j.read_text(encoding="utf-8"))
    raw = d.get("raw_response", "")
    if d.get("verdict") == "yes":
        yes_full.append((d, raw))
    else:
        no_full.append((d, raw))

print(f"Query 487b — YES count: {len(yes_full)}, NO count: {len(no_full)}")
print()
# How many YES contain explicit "VERDICT: YES" string?
explicit_yes = sum(1 for _, r in yes_full if "VERDICT: YES" in r or "VERDICT:YES" in r)
explicit_no  = sum(1 for _, r in yes_full if "VERDICT: NO" in r or "VERDICT:NO" in r)
think_only   = sum(1 for _, r in yes_full if "<think>" in r and "VERDICT:" not in r)
truncated    = sum(1 for _, r in yes_full if r.endswith("...") or len(r) < 50)
print(f"YES verdicts that contain literal 'VERDICT: YES': {explicit_yes}/{len(yes_full)}")
print(f"YES verdicts that contain literal 'VERDICT: NO' (default-YES contradicts the LLM!): {explicit_no}")
print(f"YES verdicts that have <think> but NO 'VERDICT:' line at all: {think_only}")
print(f"YES verdicts that look truncated/short: {truncated}")
print()

print("=" * 70)
print("3 RANDOM YES RAW RESPONSES IN FULL")
print("=" * 70)
import random
random.seed(0)
for d, raw in random.sample(yes_full, min(3, len(yes_full))):
    print(f"\n--- verdict={d.get('verdict')}  category={d.get('category')} ---")
    print(repr(raw))  # show escapes
    print()
