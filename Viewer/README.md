# Viewer

Simple interactive viewer for ARC tasks in this repo.

## What it shows

- Original ARC demonstrations (train pairs)
- Original ARC test inputs
- Stable synthetic pairs from:
  - `re_arc.zip` (per-task synthetic pairs)
  - `arc_gen_stable.zip`
- One dynamic synthetic example from each generator (if available)
- Verifier availability (primary: re_arc, secondary: NeurIPS Code Golf solutions)

## Run

From the repo root (`PotARCin`):

```bash
pip install -r requirements.txt
streamlit run Viewer/app.py
```

