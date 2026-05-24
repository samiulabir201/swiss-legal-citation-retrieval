"""Dump notebook code cells + truncated outputs."""
import json, sys, os, datetime

def trunc(s, lim=1500):
    if len(s) <= lim:
        return s
    return s[:800] + "\n...[truncated]...\n" + s[-400:]

def output_text(outs):
    chunks = []
    for o in outs:
        ot = o.get('output_type')
        if ot == 'stream':
            t = o.get('text', '')
            if isinstance(t, list): t = ''.join(t)
            chunks.append(t)
        elif ot in ('execute_result', 'display_data'):
            data = o.get('data', {})
            if 'text/plain' in data:
                t = data['text/plain']
                if isinstance(t, list): t = ''.join(t)
                chunks.append(t)
            if 'image/png' in data or 'image/jpeg' in data:
                chunks.append('[image omitted]')
        elif ot == 'error':
            ename = o.get('ename', '')
            evalue = o.get('evalue', '')
            tb = o.get('traceback', [])
            if isinstance(tb, list):
                tb = '\n'.join(tb)
            chunks.append(f'ERROR {ename}: {evalue}\n{tb}')
    text = '\n'.join(chunks).strip()
    # Strip ANSI escape codes
    import re
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    return text

def main():
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    st = os.stat(path)
    mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')
    size = st.st_size
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    print(f"=== META path={path}")
    print(f"=== META mtime={mtime} size={size}")
    print(f"=== META total_cells={len(cells)} code_cells={len(code_cells)}")
    # Caps
    n = len(code_cells)
    if n > 50:
        indices = list(range(0, 25)) + list(range(n-5, n))
        # Add a summary placeholder for middle
        skipped = list(range(25, n-5))
        print(f"=== META code_cell_cap_applied skipped_middle={len(skipped)} (indices {25}..{n-6})")
    else:
        indices = list(range(n))
    for i in indices:
        c = code_cells[i]
        src = c.get('source', '')
        if isinstance(src, list): src = ''.join(src)
        outs = c.get('outputs', [])
        otext = output_text(outs)
        print(f"=== CELL {i} ===")
        print("--- SRC ---")
        print(trunc(src))
        print("--- OUT ---")
        print(trunc(otext) if otext else "(no output)")
    print("=== END ===")

if __name__ == '__main__':
    main()
