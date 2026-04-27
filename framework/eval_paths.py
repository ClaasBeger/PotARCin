from __future__ import annotations

from pathlib import Path


_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.4": "GPT54",
    "claude-opus-4-6": "ClaudeO46",
    "gemini-3.1-pro-preview": "Gemini31Pro",
    "moonshotai/Kimi-K2.5": "KimiK25",
    "MiniMaxAI/MiniMax-M2.5": "MiniMaxM25",
}


def model_alias(model: str) -> str:
    m = model.strip()
    if m in _MODEL_ALIASES:
        return _MODEL_ALIASES[m]
    return "".join(ch for ch in m if ch.isalnum()) or "UnknownModel"


def run_bucket(run_tag: str) -> str:
    return "Run2" if "run2" in run_tag.lower() else "Run1"


def dimension_out_dirs(*, root_dir: Path, model: str, run_tag: str) -> dict[str, Path]:
    leaf = Path(model_alias(model)) / run_bucket(run_tag) / run_tag
    return {
        "definition": root_dir / "evaluation" / "definition_eval_results" / leaf,
        "classification_distribution": root_dir / "evaluation" / "classification_eval_results" / leaf,
        "constrained_generation": root_dir / "evaluation" / "constrained_generation_eval_results" / leaf,
        "editing": root_dir / "evaluation" / "editing_eval_results" / leaf,
        "inversion": root_dir / "evaluation" / "inversion_eval_results" / leaf,
    }


def multi_run_summary_dir(*, root_dir: Path, model: str, run_tag: str) -> Path:
    return (
        root_dir
        / "evaluation"
        / "multi_dimension_eval_runs"
        / model_alias(model)
        / run_bucket(run_tag)
        / run_tag
    )


def output_grid_run_dir(*, root_dir: Path, model: str, run_tag: str) -> Path:
    return (
        root_dir
        / "evaluation"
        / "output_grid_eval_results"
        / model_alias(model)
        / run_bucket(run_tag)
        / run_tag
    )
