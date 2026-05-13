"""Local-machine Google embedding pass.

Extends similarity_google.json in place with gemma-4 utterences + cosine scores.
Resumable: rows whose gemma keys are already present are skipped.
"""

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

ROOT = Path(__file__).parent

GEMMA_INPUTS: List[Tuple[Path, str]] = [
    (ROOT / "gemma4/model_out/checkpoint_Translation_gemma-4-26B-A4B-it.json",
     "google_gemma-4-26B-A4B-it"),
    (ROOT / "gemma4/model_out/Translation_gemma-4-E2B-it.json",
     "google_gemma-4-E2B-it"),
    (ROOT / "gemma4/model_out/Translation_gemma-4-E4B-it.json",
     "google_gemma-4-E4B-it"),
    (ROOT / "gemma4/model_out/Translation_thinking_gemma-4-E2B-it.json",
     "google_gemma-4-thinking-E2B-it"),
    (ROOT / "gemma4/model_out/Translation_thinking_gemma-4-E4B-it.json",
     "google_gemma-4-thinking-E4B-it"),
]

GOOGLE_PATH = ROOT / "Outputs" / "Similarity Test" / "similarity_google.json"
CHECKPOINT_EVERY = 100
MODEL_GOOGLE = "google/embeddinggemma-300m"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_gemma_rows(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def gemma_text_for_index(rows: List[Dict[str, Any]], idx: int) -> Optional[str]:
    if idx >= len(rows):
        return None
    row = rows[idx]
    if not isinstance(row, dict):
        return None
    ut = row.get("utterence")
    return None if ut is None else str(ut)


def cosine_normalized(a: Tensor, b: Tensor) -> float:
    return float(torch.dot(a, b).item())


def encode_google(text: str, model: Any) -> Tensor:
    emb = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    t = torch.as_tensor(emb, dtype=torch.float32).flatten()
    norm = t.norm(p=2)
    if norm.item() > 0:
        t = t / norm
    return t.cpu()


def main() -> None:
    from sentence_transformers import SentenceTransformer

    gemma_rows: List[Tuple[List[Dict[str, Any]], str]] = []
    for path, key in GEMMA_INPUTS:
        rows = load_gemma_rows(path)
        gemma_rows.append((rows, key))
        print(f"  {key}: {len(rows)} rows  ({path.name})")
    gemma_keys = [k for _, k in gemma_rows]

    records = load_json(GOOGLE_PATH)
    print(f"\nLoaded {len(records)} existing rows from {GOOGLE_PATH.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_GOOGLE, device=device)
    print(f"Loaded {MODEL_GOOGLE} on {device}\n")

    try:
        for idx in tqdm(range(len(records)), desc="Google", unit="sample"):
            row = records[idx]
            if not isinstance(row, dict):
                continue

            similarity = row.get("similarity")
            if not isinstance(similarity, dict):
                similarity = {}
                row["similarity"] = similarity
            utterence = row.get("utterence")
            if not isinstance(utterence, dict):
                utterence = {}
                row["utterence"] = utterence

            if all(k in similarity for k in gemma_keys):
                continue

            input_text = row.get("input", "")
            if not isinstance(input_text, str):
                input_text = "" if input_text is None else str(input_text)

            input_emb: Optional[Tensor] = None
            try:
                input_emb = encode_google(input_text, model)
            except Exception as exc:
                print(f"[warn] Google: input embed failed at row {idx}: {exc}")

            for rows, key in gemma_rows:
                if key in similarity:
                    continue
                translation = gemma_text_for_index(rows, idx)
                if translation is None:
                    utterence[key] = None
                    similarity[key] = None
                    continue
                utterence[key] = translation
                if input_emb is None or translation == "":
                    similarity[key] = None
                    continue
                try:
                    ut_emb = encode_google(translation, model)
                    similarity[key] = cosine_normalized(input_emb, ut_emb)
                except Exception as exc:
                    print(f"[warn] Google: utterence embed failed at row {idx}, key '{key}': {exc}")
                    similarity[key] = None

            if (idx + 1) % CHECKPOINT_EVERY == 0:
                save_json_atomic(records, GOOGLE_PATH)

        save_json_atomic(records, GOOGLE_PATH)
        print(f"\nSaved {GOOGLE_PATH}")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
