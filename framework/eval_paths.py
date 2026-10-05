from __future__ import annotations

from pathlib import Path


_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.4": "GPT54",
    "claude-opus-4-6": "ClaudeO46",
    "gemini-3.1-pro-preview": "Gemini31Pro",
    "moonshotai/Kimi-K2.5": "KimiK25",
    # Same output tree as base Kimi so :nitro resumes/skips existing run-tag results.
    "moonshotai/Kimi-K2.5:nitro": "KimiK25",
    "MiniMaxAI/MiniMax-M2.5": "MiniMaxM25",
    "minimax/minimax-m2.5:nitro": "MiniMaxM25",
}


def model_alias(model: str) -> str:
    m = model.strip()
    if m in _MODEL_ALIASES:
        return _MODEL_ALIASES[m]
    # OpenRouter speed suffixes (:nitro, etc.) should share the base model's folder.
    if ":" in m:
        base = m.rsplit(":", 1)[0]
        if base in _MODEL_ALIASES:
            return _MODEL_ALIASES[base]
    return "".join(ch for ch in m if ch.isalnum()) or "UnknownModel"


def run_bucket(run_tag: str) -> str:
    return "Run2" if "run2" in run_tag.lower() else "Run1"


def dimension_out_dirs(
    *, root_dir: Path, model: str, run_tag: str, dataset_segment: str | None = None
) -> dict[str, Path]:
    leaf = Path(model_alias(model))
    if dataset_segment:
        leaf = leaf / dataset_segment
    leaf = leaf / run_bucket(run_tag) / run_tag
    return {
        "definition": root_dir / "evaluation" / "definition_eval_results" / leaf,
        "classification_distribution": root_dir / "evaluation" / "classification_eval_results" / leaf,
        "constrained_generation": root_dir / "evaluation" / "constrained_generation_eval_results" / leaf,
        "editing": root_dir / "evaluation" / "editing_eval_results" / leaf,
        "inversion": root_dir / "evaluation" / "inversion_eval_results" / leaf,
    }


def multi_run_summary_dir(
    *, root_dir: Path, model: str, run_tag: str, dataset_segment: str | None = None
) -> Path:
    p = root_dir / "evaluation" / "multi_dimension_eval_runs" / model_alias(model)
    if dataset_segment:
        p = p / dataset_segment
    return p / run_bucket(run_tag) / run_tag


def output_grid_run_dir(
    *, root_dir: Path, model: str, run_tag: str, dataset_segment: str | None = None
) -> Path:
    p = root_dir / "evaluation" / "output_grid_eval_results" / model_alias(model)
    if dataset_segment:
        p = p / dataset_segment
    return p / run_bucket(run_tag) / run_tag
