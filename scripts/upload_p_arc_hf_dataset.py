#!/usr/bin/env python3
"""Upload the P-ARC CSV export + companion files to a Hugging Face **dataset** repo.

Stages (temp dir):
  - copies ``datasets/p_arc_hub_bundle/README.md`` and ``SCHEMA.json``
  - copies ``evaluation/figures/p_arc_dataset.csv`` (must exist — run
    ``scripts/export_p_arc_csv_dataset.py`` first)

Requires ``huggingface_hub``::

    pip install -r requirements_hf_upload.txt

Authenticate with either ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` or ``huggingface-cli login``.

Example::

    export HF_TOKEN=hf_...
    python scripts/upload_p_arc_hf_dataset.py --repo-id YourOrg/p-arc-csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _repo_root()
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        print(
            "Missing huggingface_hub. Install with:\n"
            "  pip install -r requirements_hf_upload.txt",
            file=sys.stderr,
        )
        return 1

    ap = argparse.ArgumentParser(description="Upload P-ARC CSV bundle to Hugging Face Hub.")
    ap.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id, e.g. username/p-arc-csv",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=root / "evaluation" / "figures" / "p_arc_dataset.csv",
        help="Source CSV path (default: evaluation/figures/p_arc_dataset.csv)",
    )
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=root / "datasets" / "p_arc_hub_bundle",
        help="Directory containing README.md + SCHEMA.json",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist yet.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only stage files and print paths; do not upload.",
    )
    args = ap.parse_args()

    csv_path: Path = args.csv.resolve()
    bundle: Path = args.bundle_dir.resolve()

    if not csv_path.is_file():
        print(
            f"CSV not found: {csv_path}\n"
            "Run: python scripts/export_p_arc_csv_dataset.py",
            file=sys.stderr,
        )
        return 1
    readme = bundle / "README.md"
    schema = bundle / "SCHEMA.json"
    if not readme.is_file():
        print(f"Missing README.md in bundle: {readme}", file=sys.stderr)
        return 1
    if not schema.is_file():
        print(f"Missing SCHEMA.json in bundle: {schema}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    with tempfile.TemporaryDirectory(prefix="p_arc_hf_upload_") as td_s:
        td = Path(td_s)
        shutil.copy2(readme, td / "README.md")
        shutil.copy2(schema, td / "SCHEMA.json")
        shutil.copy2(csv_path, td / "p_arc_dataset.csv")

        print("Staging directory:", td)
        for name in sorted(p.name for p in td.iterdir()):
            sz = (td / name).stat().st_size
            print(f"  {name:24} {sz:,} bytes")

        if args.dry_run:
            print("--dry-run: skipping Hub upload.")
            return 0

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(td),
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            ignore_patterns=[],
        )

    print(f"Uploaded dataset files to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
