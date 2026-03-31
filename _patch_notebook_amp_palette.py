import json
from pathlib import Path

p = Path('chip_model_unified_flow.ipynb')
nb = json.load(p.open('r', encoding='utf-8'))
changed = 0
for c in nb.get('cells', []):
    if c.get('cell_type') != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'amp_palette = {' in src and "'amp_mid'" in src:
        src = src.replace("    0.02: PAPER_COLORS['amp_low'],\n    0.05: PAPER_COLORS['amp_mid'],\n    0.10: PAPER_COLORS['amp_high'],\n", "    0.02: PAPER_COLORS.get('amp_low', PAPER_COLORS['base']),\n    0.05: PAPER_COLORS.get('amp_mid', PAPER_COLORS['accent']),\n    0.10: PAPER_COLORS.get('amp_high', PAPER_COLORS['warn']),\n")
        c['source'] = src.splitlines(keepends=True)
        changed += 1

json.dump(nb, p.open('w', encoding='utf-8', newline='\n'), indent=1)
print('cells_patched', changed)
