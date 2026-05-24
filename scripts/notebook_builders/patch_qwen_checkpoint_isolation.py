import json
import shutil
import sys
from pathlib import Path


def cell_text(cell: dict) -> str:
    return ''.join(cell.get('source', []))


def set_cell_text(cell: dict, text: str) -> None:
    cell['source'] = text.splitlines(keepends=True)


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding='utf-8'))
    changed = False

    # Patch the config cell without touching throughput/model settings.
    for cell in nb.get('cells', []):
        src = cell_text(cell)
        if "INPUT_FILE      = TARGET_CARD_FILE if ENRICH_TARGET_CARD_FILE_ONLY else ART_DIR / 'court_authority_cards_v4.jsonl'" not in src:
            continue
        old = (
            "FAILED_FILE     = ART_DIR / 'court_authority_cards_rag_failed_cards.jsonl'\n"
            "FAILED_RETRY_FILE = ART_DIR / 'court_authority_cards_rag_failed_cards_recovered.jsonl'\n"
            "FAILED_STILL_FILE = ART_DIR / 'court_authority_cards_rag_failed_cards_still_failed.jsonl'\n"
            "CHECKPOINT_FILE = ART_DIR / 'rag_checkpoint.txt'\n"
        )
        new = (
            "FAILED_FILE     = ART_DIR / ('court_authority_cards_rag_targets_failed_cards.jsonl' if ENRICH_TARGET_CARD_FILE_ONLY else 'court_authority_cards_rag_failed_cards.jsonl')\n"
            "FAILED_RETRY_FILE = ART_DIR / ('court_authority_cards_rag_targets_failed_cards_recovered.jsonl' if ENRICH_TARGET_CARD_FILE_ONLY else 'court_authority_cards_rag_failed_cards_recovered.jsonl')\n"
            "FAILED_STILL_FILE = ART_DIR / ('court_authority_cards_rag_targets_failed_cards_still_failed.jsonl' if ENRICH_TARGET_CARD_FILE_ONLY else 'court_authority_cards_rag_failed_cards_still_failed.jsonl')\n"
            "CHECKPOINT_FILE = ART_DIR / ('rag_targets_checkpoint.txt' if ENRICH_TARGET_CARD_FILE_ONLY else 'rag_checkpoint.txt')\n"
        )
        if old in src:
            src = src.replace(old, new)
            changed = True
        elif "CHECKPOINT_FILE = ART_DIR / 'rag_checkpoint.txt'" in src:
            raise RuntimeError('Found old checkpoint line but config block shape was unexpected; refusing partial patch.')

        if "print('CHECKPOINT_FILE:', CHECKPOINT_FILE)" not in src:
            src = src.replace(
                "print('FAILED_FILE:', FAILED_FILE)\n",
                "print('FAILED_FILE:', FAILED_FILE)\nprint('CHECKPOINT_FILE:', CHECKPOINT_FILE)\n",
            )
            changed = True
        set_cell_text(cell, src)
        break

    # Patch the enrichment cell so stale checkpoints cannot skip a compact input.
    for cell in nb.get('cells', []):
        src = cell_text(cell)
        if "print(f'Resuming at line {start:,}')" not in src or "target_total = min(total_lines, start + LIMIT) if LIMIT else total_lines" not in src:
            continue

        old = (
            "print(f'Resuming at line {start:,}')\n"
            "print(f'Counting lines in {INPUT_FILE.name} ...')\n"
            "\n"
            "total_lines = count_lines(INPUT_FILE)\n"
            "target_total = min(total_lines, start + LIMIT) if LIMIT else total_lines\n"
        )
        new = (
            "print(f'Checkpoint line before validation: {start:,}')\n"
            "print(f'Counting lines in {INPUT_FILE.name} ...')\n"
            "\n"
            "total_lines = count_lines(INPUT_FILE)\n"
            "if start >= total_lines and total_lines > 0:\n"
            "    print(f'[checkpoint] stale checkpoint {start:,} >= input lines {total_lines:,}; resetting to 0 for {INPUT_FILE.name}')\n"
            "    start = 0\n"
            "    CHECKPOINT_FILE.unlink(missing_ok=True)\n"
            "target_total = min(total_lines, start + LIMIT) if LIMIT else total_lines\n"
            "print(f'Resuming at validated line {start:,}')\n"
        )
        if old in src:
            src = src.replace(old, new)
            changed = True
        elif "[checkpoint] stale checkpoint" not in src:
            raise RuntimeError('Enrichment checkpoint block shape was unexpected; refusing partial patch.')
        set_cell_text(cell, src)
        break

    if not changed:
        raise RuntimeError('Notebook did not require patching or expected cells were not found.')

    backup = path.with_suffix(path.suffix + '.backup_before_checkpoint_isolation.ipynb')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Patched: {path}')
    print(f'Backup:  {backup}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/patch_qwen_checkpoint_isolation.py <notebook.ipynb>')
    patch_notebook(Path(sys.argv[1]))
