"""Repair the 66 failed JSON descriptors.

Strategy:
1. For each failure, take the LLM's raw_output (which is mostly correct but malformed).
2. Apply automatic repair steps:
   - Fix `{"de":"X":"Y"}` -> `{"de":"X","en":"Y"}` (colon-instead-of-comma typo)
   - Fix `"specificity_score": "0.8"` -> `"specificity_score": 0.8` (numeric strings)
   - Truncate at last valid object/array boundary; fill missing required fields with defaults.
3. Validate the result; if it parses, build the enrichment object.

Required schema (from existing successful records):
- english_summary: str
- legal_rule: str
- applicability_conditions: list[str]
- exceptions_or_limitations: list[str]
- legal_question: str
- concepts_en: list[str]
- terms_de_to_en: list[{de,en}]
- defined_terms: list[{term,definition}]
- addressees: list[str]
- sanctions_or_consequences: list[str]
- provision_role_llm: str
- specificity_score: float
"""
import json
import re

FAILURES_TXT = r"E:\swiss_citation_extraction\failures_with_text.jsonl"
REPAIRED = r"E:\swiss_citation_extraction\failures_repaired.jsonl"
UNREPAIRED = r"E:\swiss_citation_extraction\failures_unrepaired.jsonl"

REQUIRED_KEYS = {
    "english_summary": "",
    "legal_rule": "",
    "applicability_conditions": [],
    "exceptions_or_limitations": [],
    "legal_question": "",
    "concepts_en": [],
    "terms_de_to_en": [],
    "defined_terms": [],
    "addressees": [],
    "sanctions_or_consequences": [],
    "provision_role_llm": "other",
    "specificity_score": 0.5,
}


def fix_de_en_colon(s: str) -> str:
    # Pattern C: {"de":"X":"en":"Y"} -- two colon errors in a row, key already named "en"
    # Replace `"de":"X":"en":"Y"` with `"de":"X","en":"Y"`
    s = re.sub(
        r'("de"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*:\s*("en"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")',
        r'\1,\2',
        s,
    )
    # Pattern A: terminated by '}' -- {"de":"X":"Y"} -> {"de":"X","en":"Y"}
    s = re.sub(
        r'(\{\s*"de"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*:\s*("[^"\\]*(?:\\.[^"\\]*)*"\s*\})',
        r'\1,"en":\2',
        s,
    )
    # Pattern B: terminated by ',' -- {"de":"X":"Y","en":"Y"} -> {"de":"X","en":"Y","en":"Y"}
    s = re.sub(
        r'(\{\s*"de"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*:\s*("[^"\\]*(?:\\.[^"\\]*)*")(\s*,)',
        r'\1,"en":\2\3',
        s,
    )
    # term/definition variants
    s = re.sub(
        r'("term"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*:\s*("definition"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")',
        r'\1,\2',
        s,
    )
    s = re.sub(
        r'(\{\s*"term"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*:\s*("[^"\\]*(?:\\.[^"\\]*)*"\s*\})',
        r'\1,"definition":\2',
        s,
    )
    s = re.sub(
        r'(\{\s*"term"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*:\s*("[^"\\]*(?:\\.[^"\\]*)*")(\s*,)',
        r'\1,"definition":\2\3',
        s,
    )
    return s


def _walk_array_value(s: str, i: int):
    """Given s[i] == '[', walk forward respecting strings and bracket nesting,
    but tolerating that the LLM may have used the wrong closing bracket. Stop at
    the first depth-0 character that is `,` or `]` or `}` AND the next non-space
    character is `,\\n  "<word>"\\s*:` (a sibling top-level key) or end-of-object.

    Returns (end_index_exclusive_of_array_close, raw_array_substring).
    """
    n = len(s)
    j = i + 1
    in_str = False
    esc = False
    depth = 1
    while j < n and depth > 0:
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            j += 1
            continue
        if c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                # If next non-space, non-newline char is `,`+key or `}`,
                # we close here.
                k = j + 1
                while k < n and s[k] in " \t\r\n":
                    k += 1
                if k >= n or s[k] in ",}":
                    return j, s[i : j + 1]
                # else: false close, continue (the LLM put a wrong bracket)
                depth = 1
        j += 1
    return j - 1, s[i:j]


def _fix_array_of_pairs(arr: str, key_a: str, key_b: str) -> str:
    """Fix the value of an array-of-pair-dicts field. Replace ill-formed item
    delimiters: items may open with `[` instead of `{` and close with `]` instead
    of `}`, and the outer array may close with `}` instead of `]`.
    """
    # Inside arr (which starts with [ ... ]), normalize:
    body = arr.strip()
    # Strip outer bracket
    if not (body.startswith("[")):
        return arr
    inner = body[1:-1]  # might end with ] or }
    # Items: split on top-level commas
    items = []
    buf = ""
    depth = 0
    in_str = False
    esc = False
    for c in inner:
        if in_str:
            buf += c
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            buf += c
            continue
        if c in "[{":
            depth += 1
            buf += c
            continue
        if c in "]}":
            depth -= 1
            buf += c
            continue
        if c == "," and depth == 0:
            items.append(buf.strip())
            buf = ""
            continue
        buf += c
    if buf.strip():
        items.append(buf.strip())

    fixed_items = []
    for it in items:
        if not it:
            continue
        # Trim wrong opener
        if it.startswith("["):
            it = "{" + it[1:]
        if it.endswith("]"):
            it = it[:-1] + "}"
        fixed_items.append(it)

    return "[" + ",".join(fixed_items) + "]"


_ARRAY_PAIR_FIELDS = ["terms_de_to_en", "defined_terms"]


def fix_array_brackets_for_objects(s: str) -> str:
    """Targeted fix: only operate on the values of array-of-pair-dict fields
    (terms_de_to_en and defined_terms). For each such field, parse out the
    array value tolerantly, then reformat it with correct `{...}` delimiters.
    """
    out = s
    for field in _ARRAY_PAIR_FIELDS:
        idx = 0
        new_parts = []
        while True:
            m = re.search(r'"' + field + r'"\s*:\s*\[', out[idx:])
            if not m:
                new_parts.append(out[idx:])
                break
            start_in_out = idx + m.end() - 1  # position of '['
            # Walk forward to find array end, tolerantly
            end, arr_text = _walk_array_value(out, start_in_out)
            fixed_arr = _fix_array_of_pairs(arr_text, "de", "en") if field == "terms_de_to_en" else _fix_array_of_pairs(arr_text, "term", "definition")
            new_parts.append(out[idx:start_in_out])
            new_parts.append(fixed_arr)
            idx = end + 1
        out = "".join(new_parts)
    return out


def fix_quoted_score(s: str) -> str:
    # "specificity_score": "0.8" -> "specificity_score": 0.8
    return re.sub(
        r'("specificity_score"\s*:\s*)"((?:0(?:\.\d+)?|1(?:\.0+)?))"',
        r"\1\2",
        s,
    )


def fix_stray_paren_after_string(s: str) -> str:
    # `"ETH")}` -> `"ETH"}`  (strip stray ')' between a closing quote and '}' or ',')
    return re.sub(r'(")\)(?=\s*[,\}])', r"\1", s)


def find_balanced_object(s: str):
    """Try to extract a balanced top-level JSON object starting at first {.
    Returns (parsed, end_index) or (None, None) if no balanced object exists.
    """
    start = s.find("{")
    if start < 0:
        return None, None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                snippet = s[start : i + 1]
                try:
                    return json.loads(snippet), i + 1
                except Exception:
                    return None, None
    return None, None


def truncate_to_last_complete_field(s: str) -> str:
    """The raw_output may be truncated mid-field. Trim to a clean state where we
    can append closing braces.

    Strategy: scan once tracking depth. Record:
    - last_top_complete_pos: position after the last comma at depth 1 (so that
      truncating there gives a string of complete key:value pairs only).
    - last_array_complete_pos and array_depth_close: a snapshot of the latest
      moment when we just finished an element inside a depth-2 array (e.g.,
      an item of terms_de_to_en). Closing arrays/objects at that point
      preserves the partial array.
    """
    start = s.find("{")
    if start < 0:
        return s
    i = start + 1
    n = len(s)
    last_top_complete_pos = i  # right after opening '{'
    # snapshot info for partial arrays
    best_partial = None  # (truncate_pos, close_string)

    # track stack of '{' or '['
    stack = ["{"]
    in_str = False
    esc = False
    while i < n and stack:
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                break
            opener = stack.pop()
            if not stack:
                # whole top-level object closed cleanly
                return s[: i + 1]
            # if just closed an array element while inside a depth-2 array
            # (i.e., stack now ends with an array opener)
            if stack and stack[-1] == "[":
                # snapshot: we've just finished an element of a top-level array
                # truncate at i+1 and close: array, then top object
                trunc = s[: i + 1] + "]"
                # close any further open scopes
                # everything else above [stack-of-[ is the outer top-level {
                # so we need len(stack) - 1 closing braces ('}')
                # Actually the array opener itself is at stack[-1], so we just closed one
                # extra ']' above. Then close remaining items.
                # Note: stack still includes the array's '['.
                extra = ""
                for opener in reversed(stack[:-1]):  # all except the array we are closing
                    extra += "}" if opener == "{" else "]"
                close_str = "]" + extra
                # Replace the trunc construction: append close after position i+1
                best_partial = (i + 1, close_str)
        elif c == "," and len(stack) == 1:
            last_top_complete_pos = i  # after this comma we'd be ready for next key
        i += 1

    # Truncate to last top-level comma OR partial-array snapshot, whichever is later
    pieces = []
    if best_partial:
        pieces.append(("partial_array", best_partial[0], best_partial[1]))
    # last_top_complete strategy
    top_trunc_pos = last_top_complete_pos
    pieces.append(("top_complete", top_trunc_pos, "}"))

    # Pick the strategy that yields a parseable JSON, preferring the one that
    # keeps more content (later cutoff). Build candidates and try each.
    pieces.sort(key=lambda p: -p[1])  # later cutoff first
    for kind, pos, close in pieces:
        body = s[:pos].rstrip()
        if body.endswith(","):
            body = body[:-1]
        candidate = body + close
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return candidate
        except Exception:
            continue

    # Fallback: original simple version
    truncated = s[:last_top_complete_pos]
    truncated = truncated.rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1]
    return truncated + "\n}"


def repair_raw_output(raw: str):
    """Try multiple repair strategies. Returns parsed dict or None."""
    if not raw:
        return None

    candidates = [raw]
    candidates.append(fix_de_en_colon(raw))
    candidates.append(fix_quoted_score(raw))
    candidates.append(fix_quoted_score(fix_de_en_colon(raw)))
    candidates.append(fix_array_brackets_for_objects(raw))
    candidates.append(fix_array_brackets_for_objects(fix_de_en_colon(raw)))
    candidates.append(fix_array_brackets_for_objects(fix_quoted_score(fix_de_en_colon(raw))))
    candidates.append(fix_stray_paren_after_string(fix_array_brackets_for_objects(fix_quoted_score(fix_de_en_colon(raw)))))

    for cand in candidates:
        parsed, _ = find_balanced_object(cand)
        if isinstance(parsed, dict):
            return parsed

    # Try truncation-based repair
    for cand in [
        raw,
        fix_de_en_colon(raw),
        fix_quoted_score(raw),
        fix_quoted_score(fix_de_en_colon(raw)),
        fix_array_brackets_for_objects(fix_quoted_score(fix_de_en_colon(raw))),
    ]:
        truncated = truncate_to_last_complete_field(cand)
        try:
            parsed = json.loads(truncated)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def normalize(parsed: dict) -> dict:
    """Make sure all required keys exist and are valid types."""
    out = {}
    for k, default in REQUIRED_KEYS.items():
        v = parsed.get(k, default)
        if isinstance(default, list):
            if not isinstance(v, list):
                v = []
            # normalize items
            if k == "terms_de_to_en":
                cleaned = []
                for item in v:
                    if isinstance(item, dict) and "de" in item and "en" in item:
                        cleaned.append({"de": str(item["de"]), "en": str(item["en"])})
                v = cleaned
            elif k == "defined_terms":
                cleaned = []
                for item in v:
                    if isinstance(item, dict) and "term" in item and "definition" in item:
                        cleaned.append({"term": str(item["term"]), "definition": str(item["definition"])})
                v = cleaned
            else:
                v = [str(x) for x in v if x is not None]
        elif isinstance(default, str):
            if v is None:
                v = ""
            else:
                v = str(v)
        elif isinstance(default, (float, int)):
            try:
                v = float(v)
            except Exception:
                v = float(default)
        out[k] = v
    return out


def main():
    repaired = []
    unrepaired = []

    with open(FAILURES_TXT, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec.get("raw_output") or ""
            parsed = repair_raw_output(raw)
            if parsed is None:
                unrepaired.append(rec)
                continue
            enrichment = normalize(parsed)
            rec["repaired_enrichment"] = enrichment
            repaired.append(rec)

    with open(REPAIRED, "w", encoding="utf-8") as f:
        for r in repaired:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(UNREPAIRED, "w", encoding="utf-8") as f:
        for r in unrepaired:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Repaired: {len(repaired)}")
    print(f"Unrepaired: {len(unrepaired)}")
    if unrepaired:
        print("Unrepaired source rows:", [r["_source_row"] for r in unrepaired])


if __name__ == "__main__":
    main()
