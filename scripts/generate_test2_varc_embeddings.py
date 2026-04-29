#!/usr/bin/env python3
"""Generate Test2 embeddings by mirroring VARC reference protocol.

Reference-matched behavior:
- Build per-task split: train_color_permute_ttt_9/<task_id>
- Variants: original + 5 basic transforms x (identity + 9 color perms) => 51 files
- Two-stage TTT: phase1 (task token only), then phase2 (model only)
- Save embedding payload in the same dict schema as external/Task_Embeddings
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
VARC_ROOT = REPO_ROOT / "external" / "VARC" / "VARC"

import sys

if str(VARC_ROOT) not in sys.path:
    sys.path.insert(0, str(VARC_ROOT))

from src.ARC_ViT import ARCViT  # noqa: E402
from src.ARC_loader import ARCDataset, IGNORE_INDEX, collate_fn  # noqa: E402
from utils.data_augmentation import augment_raw_data_split_per_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test2-root",
        type=Path,
        default=REPO_ROOT / "Test2",
        help="Root directory containing Test2 task json files.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="**/*.json",
        help="Glob pattern under test2-root for selecting task files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "external" / "VARC" / "saves" / "offline_train_ViT" / "checkpoint_best.pt",
        help="Path to VARC offline-trained ViT checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "external" / "Task_Embeddings",
        help="Directory where *_task_token_embed.pt files are written.",
    )
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-colors", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of tasks to process (0 = all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose embedding file already exists.",
    )
    parser.add_argument(
        "--save-stage-label",
        type=str,
        choices=("phase1", "phase2"),
        default="phase1",
        help="Which stage embedding to export (reference embeddings use phase1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Compute device. 'auto' picks cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--disable-translation",
        action="store_true",
        help="Disable random translation augmentation.",
    )
    parser.add_argument(
        "--disable-resolution-augmentation",
        action="store_true",
        help="Disable random resolution augmentation.",
    )
    parser.add_argument(
        "--fix-scale-factor",
        type=int,
        default=1,
        help="Scale factor used when resolution augmentation is disabled.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_task_files(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _prepare_reference_split(task_file: Path, tmp_root: Path) -> str:
    """Create train_color_permute_ttt_9/<task_id> exactly like reference."""
    dataset_training = tmp_root / "data" / "training"
    dataset_training.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_file, dataset_training / task_file.name)

    augment_raw_data_split_per_task(
        dataset_root=tmp_root,
        split="training",
        output_subdir="train_color_permute_ttt_9",
        num_permuate=9,
        only_basic=True,
        verbose=False,
    )
    return f"train_color_permute_ttt_9/{task_file.stem}"


def build_loader_for_reference_split(
    split_name: str,
    *,
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    disable_translation: bool,
    disable_resolution_augmentation: bool,
    fix_scale_factor: int,
) -> tuple[ARCDataset, DataLoader]:
    dataset = ARCDataset(root=data_root, split=split_name, subset="train", max_size=image_size)
    if disable_translation:
        dataset.disable_translation()
    if disable_resolution_augmentation:
        dataset.disable_resolution_augmentation(fix_scale_factor=fix_scale_factor)
    else:
        dataset.enable_resolution_augmentation()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataset, loader


def load_model_from_checkpoint(args: argparse.Namespace, device: torch.device, num_tasks: int) -> ARCViT:
    model = ARCViT(
        num_tasks=num_tasks,
        image_size=args.image_size,
        num_colors=args.num_colors,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        patch_size=args.patch_size,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    clean_state = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    clean_state.pop("task_token_embed.weight", None)  # Always re-learn task tokens at TTT
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if unexpected:
        print(f"Warning: unexpected params ignored: {sorted(unexpected)}")
    if missing and missing != ["task_token_embed.weight"]:
        print(f"Warning: missing params when loading checkpoint: {sorted(missing)}")
    return model


def run_stage(
    *,
    model: ARCViT,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    train_task_token: bool,
) -> None:
    for p in model.parameters():
        p.requires_grad = not train_task_token
    model.task_token_embed.weight.requires_grad = train_task_token

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected for stage.")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        count = 0
        for batch in loader:
            inputs = batch["inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            task_ids = batch["task_ids"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, task_ids, attention_mask=attention_mask)
            num_colors = logits.size(1)
            loss = F.cross_entropy(
                logits.permute(0, 2, 3, 1).reshape(-1, num_colors),
                targets.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            batch_size = inputs.size(0)
            running += loss.item() * batch_size
            count += batch_size
        print(f"  epoch {epoch:03d}/{epochs:03d} | loss={running / max(count, 1):.5f}")


def _save_reference_payload(
    *,
    model: ARCViT,
    dataset: ARCDataset,
    out_path: Path,
    eval_split: str,
    stage_label: str,
    task_stem: str,
    attempt_idx: int,
) -> None:
    task_lookup = dict(dataset.task_lookup)
    sorted_lookup_items = sorted(task_lookup.items(), key=lambda kv: kv[1])
    task_names = [name for name, _ in sorted_lookup_items]
    base_task_names = [task_stem]
    base_idx = task_lookup[task_stem]

    payload = {
        "task_token_embed.weight": model.task_token_embed.weight.detach().cpu()[base_idx : base_idx + 1].clone(),
        "task_lookup": task_lookup,
        "task_names": task_names,
        "base_task_names": base_task_names,
        "attempt_idx": int(attempt_idx),
        "eval_split": eval_split,
        "stage_label": stage_label,
    }
    torch.save(payload, out_path)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.test2_root.exists():
        raise FileNotFoundError(f"Test2 root not found: {args.test2_root}")

    set_seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_files = list(iter_task_files(args.test2_root, args.file_pattern))
    if args.limit > 0:
        task_files = task_files[: args.limit]
    if not task_files:
        raise RuntimeError(f"No tasks found under {args.test2_root} matching {args.file_pattern}")

    print(f"Found {len(task_files)} task files.")
    for i, task_file in enumerate(task_files, start=1):
        out_path = args.output_dir / f"{task_file.stem}_task_token_embed.pt"
        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(task_files)}] skip {task_file.name} (exists)")
            continue

        print(f"[{i}/{len(task_files)}] processing {task_file}")
        with tempfile.TemporaryDirectory(prefix="varc_ref_embed_") as tmp:
            tmp_root = Path(tmp)
            split_name = _prepare_reference_split(task_file, tmp_root)
            dataset, loader = build_loader_for_reference_split(
                split_name,
                data_root=tmp_root,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                disable_translation=args.disable_translation,
                disable_resolution_augmentation=args.disable_resolution_augmentation,
                fix_scale_factor=args.fix_scale_factor,
            )
            model = load_model_from_checkpoint(args, device, num_tasks=dataset.num_tasks)

            print(" stage 1/2: task token only")
            run_stage(
                model=model,
                loader=loader,
                device=device,
                epochs=args.stage1_epochs,
                lr=args.stage1_lr,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                train_task_token=True,
            )
            if args.save_stage_label == "phase1":
                _save_reference_payload(
                    model=model,
                    dataset=dataset,
                    out_path=out_path,
                    eval_split=split_name,
                    stage_label="phase1",
                    task_stem=task_file.stem,
                    attempt_idx=0,
                )
                print(f" saved phase1 embedding payload to {out_path}")

            print(" stage 2/2: model only (task token frozen)")
            if args.stage2_epochs > 0:
                run_stage(
                    model=model,
                    loader=loader,
                    device=device,
                    epochs=args.stage2_epochs,
                    lr=args.stage2_lr,
                    weight_decay=args.weight_decay,
                    max_grad_norm=args.max_grad_norm,
                    train_task_token=False,
                )

            if args.save_stage_label == "phase2":
                _save_reference_payload(
                    model=model,
                    dataset=dataset,
                    out_path=out_path,
                    eval_split=split_name,
                    stage_label="phase2",
                    task_stem=task_file.stem,
                    attempt_idx=0,
                )
                print(f" saved phase2 embedding payload to {out_path}")


if __name__ == "__main__":
    main()
