#!/usr/bin/env python3
"""Merge model translation outputs into one JSON file.

Output format per sample:
{
  "input": <English source from NLLB-200/5000-samples.json>,
  "utterence": {
    <model_name_1>: <model_1_utterence>,
    ...
  },
  "label": <label retained from model files>
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Global configuration
MODELS_DIR = Path("Outputs/All Models Test")
SOURCE_FILE = Path("NLLB-200/5000-samples.json")
OUTPUT_FILE = Path("Outputs/All Models Test/merged_translations.json")

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_name_from_file(path: Path) -> str:
    name = path.stem
    if name.startswith("experiment_"):
        name = name[len("experiment_") :]
    return name


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")
    if not MODELS_DIR.exists() or not MODELS_DIR.is_dir():
        raise NotADirectoryError(f"Models directory not found: {MODELS_DIR}")

    source_samples = load_json(SOURCE_FILE)
    if not isinstance(source_samples, list):
        raise ValueError(f"Expected a JSON array in source file: {SOURCE_FILE}")

    model_files = sorted(MODELS_DIR.glob("experiment_*.json"))
    if not model_files:
        raise FileNotFoundError(
            f"No model files matched 'experiment_*.json' in {MODELS_DIR}"
        )

    model_data: dict[str, list[dict[str, Any]]] = {}
    sample_count = len(source_samples)

    for model_file in model_files:
        model_name = model_name_from_file(model_file)
        data = load_json(model_file)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in model file: {model_file}")
        if len(data) != sample_count:
            raise ValueError(
                f"Length mismatch for {model_file}: expected {sample_count}, got {len(data)}"
            )

        model_data[model_name] = data

    merged: list[dict[str, Any]] = []

    for idx in range(sample_count):
        src = source_samples[idx]
        if not isinstance(src, dict):
            raise ValueError(f"Source sample at index {idx} is not an object")

        english_source = src.get("utterance")
        if english_source is None:
            raise KeyError(f"Missing 'utterance' in source sample at index {idx}")

        row: dict[str, Any] = {
            "input": english_source,
            "utterence": {},
            "label": src.get("label"),
        }

        # Keep label from model files (as requested), while preserving source label as fallback.
        for model_name in sorted(model_data.keys()):
            sample = model_data[model_name][idx]
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Sample at index {idx} in model '{model_name}' is not an object"
                )

            row["utterence"][model_name] = sample.get("utterence")
            if row["label"] is None and sample.get("label") is not None:
                row["label"] = sample.get("label")
            elif sample.get("label") is not None:
                row["label"] = sample.get("label")

        merged.append(row)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {sample_count} samples from {len(model_files)} models.")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
