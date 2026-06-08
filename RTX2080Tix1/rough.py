#!/usr/bin/env python3
"""
Sarvam-Translate En->Te on a SINGLE RTX 2080 Ti (11 GB, Turing / sm_75).

Self-contained port of Optional/translate_sarvam.py — no `ensemble_common` import, so this
folder (rough.py + wave_07.json + test_mini.json) can be dropped onto the new box as-is.

Why this differs from the repo version (the Turing card forces two hands):
  * dtype = fp16, NOT bf16. Turing has no native bfloat16; bf16 would silently run in software
    and tank throughput. (Ampere+ is where bf16 belongs.)
  * attention = sdpa (then eager) — flash_attention_2 needs sm_80+, so it is never tried here.
  * single device cuda:0, modest batch. Sarvam-Translate is Gemma3-4B (~4.3B params) => ~8.6 GB
    in fp16, leaving only ~2 GB for KV-cache + activations + CUDA context. Batch stays small and
    the run loop degrades a too-big batch to per-item on OOM instead of dying.

Contract (unchanged, so output still drops into refine_chimera.load_base()):
  read   wave json      [{uid, input_text, label, ...}]
  write  append jsonl   {uid, input_text, label, model:"sarvam", telugu}   (resumable by uid)

Quick smoke test (does the model actually load + generate?):
  python rough.py --dataset test_mini.json
Full run on the wave:
  python rough.py --dataset wave_07.json
  python rough.py --dataset wave_07.json --batch-size 8        # if VRAM allows
  python rough.py --dataset wave_07.json --limit 20            # first 20 pending only

Deps: torch (CUDA build), transformers (>=4.50 for Gemma3), tqdm, accelerate.
HF_TOKEN env var is honored if the repo ever needs auth.
"""
from __future__ import annotations

import argparse
import json
import os

# Reduce fragmentation OOMs on an 11 GB card — must be set before torch grabs the allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))   # paths resolve next to this script
TAG = "sarvam"
DEFAULT_REPO = "sarvamai/sarvam-translate"
TGT_LANG_NAME = "Telugu"


# ----------------------------------------------------------------- path / io helpers
def resolve(path: str) -> str:
    """Absolute as-is; relative is anchored next to rough.py (cwd-independent)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def read_wave(path: str) -> list[dict]:
    return json.load(open(resolve(path), encoding="utf-8"))


def done_uids(path: str) -> set[str]:
    """uids already written, so a re-run resumes instead of re-translating."""
    done: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["uid"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


# ----------------------------------------------------------------- OOM-safe run loop
def run_loop(pending, out_path, translate_fn, batch_size) -> dict:
    """Batched, append-only, OOM-degrades-to-per-item."""
    fh = open(out_path, "a", encoding="utf-8", buffering=1)
    stats = {"oom": 0, "err": 0}
    bar = tqdm(total=len(pending), desc=TAG, unit="row")

    def emit(r, te, bucket=None):
        if bucket:
            stats[bucket] += 1
        fh.write(json.dumps({"uid": r["uid"], "input_text": r["input_text"],
                             "label": r["label"], "model": TAG, "telugu": te},
                            ensure_ascii=False) + "\n")

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        try:
            hyps = translate_fn([r["input_text"] for r in batch])
            for r, te in zip(batch, hyps):
                emit(r, te)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tqdm.write(f"  [WARN] OOM on batch of {len(batch)} -> per-item")
            for r in batch:                          # isolate the offender, keep the rest
                try:
                    emit(r, translate_fn([r["input_text"]])[0])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); emit(r, "", "oom")
                    tqdm.write(f"  [WARN] OOM uid={r['uid']} -> empty")
                except Exception as e:               # noqa: BLE001
                    emit(r, "", "err")
                    tqdm.write(f"  [WARN] err uid={r['uid']}: {str(e)[:80]}")
        except Exception as e:                        # noqa: BLE001 — never let one batch kill the run
            tqdm.write(f"  [WARN] err on batch -> per-item: {str(e)[:80]}")
            for r in batch:
                try:
                    emit(r, translate_fn([r["input_text"]])[0])
                except Exception as e2:               # noqa: BLE001
                    emit(r, "", "err")
                    tqdm.write(f"  [WARN] err uid={r['uid']}: {str(e2)[:80]}")
        fh.flush()
        bar.update(len(batch))

    bar.close(); fh.close()
    print(f"\n[SUMMARY] {TAG}: pending={len(pending)} oom={stats['oom']} err={stats['err']} -> {out_path}",
          flush=True)
    return stats


# ----------------------------------------------------------------- model loading
def load_model(repo, *, dtype, device, attn_candidates, token):
    """Load `repo` with the first attention kernel that works, whole model on one device."""
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True, token=token)
    last = None
    for impl in attn_candidates:
        try:
            m = AutoModelForCausalLM.from_pretrained(repo, attn_implementation=impl, **kwargs)
            m = m.to(device)
            return m.eval(), impl
        except Exception as e:                       # unsupported kernel / arch mismatch / missing wheel
            last = e
            print(f"  [attn] {impl} unavailable ({str(e)[:80]}), trying next", flush=True)
    raise RuntimeError(f"could not load {repo}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wave_07.json",
                    help="wave json [{uid,input_text,label,...}] (rel paths anchored next to rough.py)")
    ap.add_argument("--out", default=None,
                    help="output jsonl (default: <stem>.sarvam.jsonl next to rough.py)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF id or local path")
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"],
                    help="fp16 is correct for the 2080 Ti; bf16 is emulated (slow) on Turing")
    ap.add_argument("--attn", default="auto",
                    help="attention kernel: auto (sdpa->eager) | sdpa | eager  (no flash_attn on sm_75)")
    # 2080 Ti is tight: 4 is a safe default for tweet-length inputs; OOM auto-falls back to per-item.
    ap.add_argument("--batch-size", type=int, default=4, help="rows per generate()")
    ap.add_argument("--max-input-tokens", type=int, default=512, help="source truncation length")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="max generated Telugu tokens")
    ap.add_argument("--temperature", type=float, default=0.01, help="repo default 0.01 (near-greedy)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="process at most N pending rows (smoke test)")
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if args.dtype == "bf16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        print("  [WARN] bf16 not natively supported on this GPU (Turing); it will be slow. Prefer fp16.")
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    if device == "cpu":
        dtype = torch.float32

    out_path = resolve(args.out) if args.out else resolve(
        os.path.splitext(os.path.basename(args.dataset))[0] + f".{TAG}.jsonl")
    rows = read_wave(args.dataset)
    seen = done_uids(out_path)
    pending = [r for r in rows if r["uid"] not in seen]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f"{TAG}: rows={len(rows)} done={len(seen)} pending={len(pending)} "
          f"bs={args.batch_size} device={device} dtype={args.dtype} -> {out_path}", flush=True)
    if not pending:
        print("nothing to do"); return

    if device.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {name} | {free/2**30:.1f} GB free / {total/2**30:.1f} GB total", flush=True)

    print(f"\nLoading {args.repo} (dtype={args.dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.repo, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"             # batched causal generation needs left padding
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    attn = [args.attn] if args.attn != "auto" else (["sdpa", "eager"] if device.startswith("cuda") else ["eager"])
    model, impl = load_model(args.repo, dtype=dtype, device=device, attn_candidates=attn,
                             token=os.environ.get("HF_TOKEN"))
    print(f"✓ loaded (attn={impl})\n", flush=True)

    @torch.inference_mode()
    def translate(texts: list[str]) -> list[str]:
        out = [""] * len(texts)
        keep = [i for i, t in enumerate(texts) if t and isinstance(t, str) and t.strip()]
        if not keep:
            return out
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": f"Translate the text below to {TGT_LANG_NAME}."},
                 {"role": "user", "content": texts[i].strip()}],
                tokenize=False, add_generation_prompt=True)
            for i in keep
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_input_tokens).to(model.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=args.temperature > 0, temperature=args.temperature,
                             top_p=args.top_p, num_return_sequences=1,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]      # left padding => one slice fits the whole batch
        dec = tok.batch_decode(new, skip_special_tokens=True)
        for j, i in enumerate(keep):
            out[i] = dec[j].strip()
        return out

    run_loop(pending, out_path, translate, args.batch_size)


if __name__ == "__main__":
    main()
