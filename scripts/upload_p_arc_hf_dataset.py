#!/usr/bin/env python3
"""Stage the P-ARC dataset and upload it to a Hugging Face **dataset** repo.

Stages:
  - ``datasets/p_arc_hub_bundle/README.md`` (dataset card) and ``SCHEMA.json``
  - ``evaluation/figures/p_arc_dataset.csv`` (run ``scripts/export_p_arc_csv_dataset.py`` first)
    and a Parquet copy of it, which the dataset viewer and ``load_dataset`` use
  - ``tasks/<task_id>.json`` with each task's train/test pairs in the ARC JSON format
  - ``datasets/p_arc_hub_bundle/P-ARC.croissant.json`` (Croissant metadata with the
    responsible-AI fields), with the CSV's sha256 and the repo URL filled in

Requires ``huggingface_hub``, ``pandas`` and ``pyarrow``::

    pip install -r requirements_hf_upload.txt

Authenticate with ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` or ``huggingface-cli login``.

Example::

    python scripts/upload_p_arc_hf_dataset.py --dry-run --stage-dir /tmp/p_arc_stage
    python scripts/upload_p_arc_hf_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage(csv_path: Path, bundle: Path, repo_id: str, out: Path) -> None:
    import pandas as pd

    shutil.copy2(bundle / "README.md", out / "README.md")
    shutil.copy2(bundle / "SCHEMA.json", out / "SCHEMA.json")
    shutil.copy2(csv_path, out / "p_arc_dataset.csv")

    csv.field_size_limit(sys.maxsize)
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pd.DataFrame(rows).to_parquet(out / "p_arc_dataset.parquet", index=False)

    tasks = out / "tasks"
    tasks.mkdir(exist_ok=True)
    for r in rows:
        task = {
            "train": json.loads(r["train_demonstrations_json"]),
            "test": [{"input": json.loads(r["test_input_json"]), "output": json.loads(r["test_output_json"])}],
        }
        (tasks / f"{r['task_id']}.json").write_text(json.dumps(task, separators=(",", ":")), encoding="utf-8")

    croissant = json.loads((bundle / "P-ARC.croissant.json").read_text(encoding="utf-8"))
    base = f"https://huggingface.co/datasets/{repo_id}"
    croissant["url"] = base
    croissant["alternateName"] = [repo_id]
    for dist in croissant["distribution"]:
        if dist.get("@id") == "p_arc_dataset.csv":
            dist["contentUrl"] = f"{base}/resolve/main/p_arc_dataset.csv"
            dist["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (out / "P-ARC.croissant.json").write_text(json.dumps(croissant, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Stage and upload the P-ARC dataset to the Hugging Face Hub.")
    ap.add_argument("--repo-id", default="ClaasBeger/P-ARC", help="Dataset repo id (default: ClaasBeger/P-ARC).")
    ap.add_argument(
        "--csv",
        type=Path,
        default=root / "evaluation" / "figures" / "p_arc_dataset.csv",
        help="Source CSV path (default: evaluation/figures/p_arc_dataset.csv).",
    )
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=root / "datasets" / "p_arc_hub_bundle",
        help="Directory containing README.md, SCHEMA.json and P-ARC.croissant.json.",
    )
    ap.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist yet.")
    ap.add_argument("--dry-run", action="store_true", help="Stage the files and list them; do not upload.")
    ap.add_argument("--stage-dir", type=Path, default=None, help="Stage into this directory and keep it (default: a temporary one).")
    args = ap.parse_args()

    csv_path: Path = args.csv.resolve()
    bundle: Path = args.bundle_dir.resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}\nRun: python scripts/export_p_arc_csv_dataset.py", file=sys.stderr)
        return 1
    for name in ("README.md", "SCHEMA.json", "P-ARC.croissant.json"):
        if not (bundle / name).is_file():
            print(f"Missing {name} in bundle: {bundle}", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory(prefix="p_arc_hf_upload_") as td_s:
        stage = args.stage_dir.resolve() if args.stage_dir else Path(td_s)
        stage.mkdir(parents=True, exist_ok=True)
        _stage(csv_path, bundle, args.repo_id, stage)

        print("Staging directory:", stage)
        for p in sorted(stage.iterdir()):
            if p.is_file():
                print(f"  {p.name:24} {p.stat().st_size:,} bytes")
        print(f"  tasks/                   {len(list((stage / 'tasks').glob('*.json')))} files")

        if args.dry_run:
            print("--dry-run: skipping Hub upload.")
            return 0

        from huggingface_hub import HfApi  # type: ignore[import-untyped]

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=args.repo_id, repo_type="dataset", token=token)

    print(f"Uploaded dataset files to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
