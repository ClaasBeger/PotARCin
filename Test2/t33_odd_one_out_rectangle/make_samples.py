from __future__ import annotations

import html
import json
import random
from pathlib import Path

from generator import generate
from verifier import verify_t33

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / 't33_samples_50.json'
HTML_PATH = ROOT / 't33_samples_50.html'

COLORS = {
    0: '#000000',
    1: '#1f77b4',
    2: '#d62728',
    3: '#2ca02c',
    4: '#ffdf00',
    5: '#7f7f7f',
    6: '#ff00ff',
    7: '#ff7f0e',
    8: '#17becf',
    9: '#8c564b',
}


def render_grid(grid: list[list[int]], cell: int = 18) -> str:
    rows = []
    for row in grid:
        cells = ''.join(
            f'<td style="width:{cell}px;height:{cell}px;background:{COLORS[v]};border:1px solid #555"></td>'
            for v in row
        )
        rows.append(f'<tr>{cells}</tr>')
    return f'<table style="border-collapse:collapse">{"".join(rows)}</table>'


def main() -> None:
    rng = random.Random(0)
    samples = []
    for _ in range(50):
        ex = generate(rng=rng)
        if verify_t33(ex['input']) != ex['output']:
            raise RuntimeError('Verifier mismatch on generated sample')
        samples.append(ex)
    JSON_PATH.write_text(json.dumps({'train': samples}, indent=2))

    blocks = [
        '<html><head><meta charset="utf-8"><title>t33 samples</title></head><body>',
        '<h1>t33 odd one out rectangle — 50 samples</h1>',
    ]
    for i, ex in enumerate(samples):
        blocks.append(f'<h2>Sample {i + 1}</h2>')
        blocks.append('<div style="display:flex;gap:30px;align-items:flex-start">')
        blocks.append('<div><h3>Input</h3>' + render_grid(ex['input']) + '</div>')
        blocks.append('<div><h3>Output</h3>' + render_grid(ex['output']) + '</div>')
        blocks.append('</div>')
    blocks.append('</body></html>')
    HTML_PATH.write_text(''.join(blocks))

    print(f'Wrote {JSON_PATH}')
    print(f'Wrote {HTML_PATH}')


if __name__ == '__main__':
    main()
