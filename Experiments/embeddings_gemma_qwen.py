"""Standalone Qwen embedding pass for similarity_qwen.json.

Hardware target: 4x RTX 2080ti (Turing, no flash-attn). Two Qwen model
instances run in parallel as child processes:
  - Worker 0 -> GPUs {0, 1}
  - Worker 1 -> GPUs {2, 3}

Each instance loads Qwen3-Embedding-8B with accelerate device_map="auto" and
max_memory restricted to its 2 GPUs (9 GiB cap each) plus a 30 GiB CPU
backstop. attn_implementation="eager" disables flash-attn (unsupported on
Turing). The 5000-row dataset is split in half between the workers.

Fresh run: existing gemma similarity keys are recomputed (not skipped).

Single command to launch:
    python embeddings_gemma_qwen.py
"""

import gc
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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

QWEN_PATH = ROOT / "Outputs" / "Similarity Test" / "similarity_qwen.json"

CHECKPOINT_EVERY = 100
MODEL_QWEN = "Qwen/Qwen3-Embedding-8B"
QWEN_MAX_LENGTH = 8192

# Two model instances, each pinned to 2 GPUs via accelerate max_memory.
WORKER_GPU_GROUPS: List[List[int]] = [[0, 1], [2, 3]]
VRAM_CAP_GIB_PER_GPU = 9
RAM_CAP_GIB = 30


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    # RTX 2080ti is Turing (sm_75) and has no native bf16. Use fp16.
    return torch.float16


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    indices = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[indices, sequence_lengths]


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for location in hf_map.values():
            if isinstance(location, int):
                return torch.device(f"cuda:{location}")
            if isinstance(location, str) and location.startswith("cuda"):
                return torch.device(location)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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


def load_qwen(gpu_ids: List[int]):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN, padding_side="left")
    max_memory: Dict[Any, str] = {gid: f"{VRAM_CAP_GIB_PER_GPU}GiB" for gid in gpu_ids}
    max_memory["cpu"] = f"{RAM_CAP_GIB}GiB"
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": preferred_dtype(),
        "device_map": "auto",
        "max_memory": max_memory,
        "attn_implementation": "eager",
    }
    model = AutoModel.from_pretrained(MODEL_QWEN, **load_kwargs)
    model.eval()
    return tokenizer, model, get_model_input_device(model)


def encode_qwen(text: str, tokenizer: Any, model: Any, input_device: torch.device) -> Tensor:
    batch = tokenizer(
        [text],
        truncation=True,
        max_length=QWEN_MAX_LENGTH,
        return_tensors="pt",
    )
    batch = {k: v.to(input_device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].float().cpu()


def worker_main(
    worker_id: int,
    gpu_ids: List[int],
    start: int,
    end: int,
    shard_path_str: str,
) -> None:
    """Child process: process rows [start, end) and write to shard_path."""
    shard_path = Path(shard_path_str)
    print(f"[w{worker_id}] starting, GPUs={gpu_ids}, slice=[{start},{end})", flush=True)

    full = load_json(QWEN_PATH)
    shard_rows = full[start:end]
    del full

    gemma_rows: List[Tuple[List[Dict[str, Any]], str]] = []
    for path, key in GEMMA_INPUTS:
        rows = load_gemma_rows(path)
        gemma_rows.append((rows, key))

    print(f"[w{worker_id}] loading {MODEL_QWEN}...", flush=True)
    tokenizer, model, input_device = load_qwen(gpu_ids)
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        unique = sorted({str(v) for v in hf_map.values()})
        print(f"[w{worker_id}] device_map locations: {unique}", flush=True)
    print(f"[w{worker_id}] input_device={input_device}", flush=True)

    try:
        bar = tqdm(
            range(len(shard_rows)),
            desc=f"w{worker_id}[{start}:{end})",
            unit="sample",
            position=worker_id,
            mininterval=10,
        )
        for local_idx in bar:
            global_idx = start + local_idx
            row = shard_rows[local_idx]
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

            input_text = row.get("input", "")
            if not isinstance(input_text, str):
                input_text = "" if input_text is None else str(input_text)

            input_emb: Optional[Tensor] = None
            try:
                input_emb = encode_qwen(input_text, tokenizer, model, input_device)
            except Exception as exc:
                print(f"[w{worker_id}] input embed failed at row {global_idx}: {exc}", flush=True)

            for grows, key in gemma_rows:
                translation = gemma_text_for_index(grows, global_idx)
                if translation is None:
                    utterence[key] = None
                    similarity[key] = None
                    continue
                utterence[key] = translation
                if input_emb is None or translation == "":
                    similarity[key] = None
                    continue
                try:
                    ut_emb = encode_qwen(translation, tokenizer, model, input_device)
                    similarity[key] = cosine_normalized(input_emb, ut_emb)
                except Exception as exc:
                    print(
                        f"[w{worker_id}] utterence embed failed at row {global_idx}, "
                        f"key '{key}': {exc}",
                        flush=True,
                    )
                    similarity[key] = None

            if (local_idx + 1) % CHECKPOINT_EVERY == 0:
                save_json_atomic(shard_rows, shard_path)

        save_json_atomic(shard_rows, shard_path)
        print(f"[w{worker_id}] done, saved {shard_path}", flush=True)
    finally:
        del tokenizer
        del model
        cleanup_memory()


def split_boundaries(total: int, n_workers: int) -> List[Tuple[int, int]]:
    chunk = total // n_workers
    out = []
    for i in range(n_workers):
        start = i * chunk
        end = (i + 1) * chunk if i < n_workers - 1 else total
        out.append((start, end))
    return out


def main() -> None:
    print(f"Reading {QWEN_PATH}")
    full = load_json(QWEN_PATH)
    total = len(full)
    print(f"  {total} rows")
    if total == 0:
        raise RuntimeError("No rows in similarity_qwen.json")

    n_workers = len(WORKER_GPU_GROUPS)
    boundaries = split_boundaries(total, n_workers)
    shard_paths = [QWEN_PATH.with_name(f".qwen_shard_w{i}.json") for i in range(n_workers)]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs: List[mp.Process] = []
    for i, ((start, end), gpu_ids, shard_path) in enumerate(
        zip(boundaries, WORKER_GPU_GROUPS, shard_paths)
    ):
        p = mp.Process(
            target=worker_main,
            args=(i, gpu_ids, start, end, str(shard_path)),
            name=f"qwen-worker-{i}",
        )
        p.start()
        procs.append(p)
        print(f"spawned worker {i} (PID {p.pid}) rows=[{start},{end}) GPUs={gpu_ids}")

    failed: List[Tuple[str, int]] = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((p.name, p.exitcode or -1))
    if failed:
        raise RuntimeError(f"Workers failed: {failed}")

    print("\nMerging shards into similarity_qwen.json...")
    merged = list(full)
    for (start, end), shard_path in zip(boundaries, shard_paths):
        shard = load_json(shard_path)
        if len(shard) != end - start:
            raise ValueError(
                f"Shard {shard_path} has {len(shard)} rows, expected {end - start}"
            )
        for i, rec in enumerate(shard):
            merged[start + i] = rec
    save_json_atomic(merged, QWEN_PATH)
    print(f"wrote {len(merged)} rows to {QWEN_PATH}")

    for shard_path in shard_paths:
        try:
            shard_path.unlink()
        except FileNotFoundError:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
