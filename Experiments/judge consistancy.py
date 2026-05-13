import gc
import json
import os
import re
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE = (
    # "/home/phani/Documents/Spring 2K26/BTP-I/Experiments/"
    "Outputs/Refinement Test/refined_translations_chimera.json"
)
OUTPUT_DIR = "/ssd_scratch/phani_ssd/model_out"
NUM_RUNS = 100
EXAMPLE_IDX = 0  # Index of the example to test for consistency

# Set your Hugging Face token to access gated models (e.g. Llama-3.1)
# Export as env var:  export HF_TOKEN="hf_xxxx"  or hard-code below.
import os as _os
HF_TOKEN: str | None = _os.environ.get("HF_TOKEN", None)
del _os

# Models to skip — either already completed 100/100 runs, or access not yet granted.
COMPLETED_MODELS = [
    # "meta-llama/Llama-3.1-8B-Instruct",   # gated — skipping until access granted
    "bigscience/bloom-7b1",               # completed run 1
    "mistralai/Mistral-7B-Instruct-v0.3", # completed run 1
    "google/gemma-7b-it",                 # completed run 1
    "Qwen/Qwen2.5-7B-Instruct",           # completed run 2
    "tiiuae/falcon-7b-instruct",          # completed run 2
    "lmsys/vicuna-13b-v1.5",              # completed run 3
]

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "lmsys/vicuna-13b-v1.5",
    "Qwen/Qwen2.5-7B-Instruct",
    "bigscience/bloom-7b1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-7b-it",
    "tiiuae/falcon-7b-instruct",
]

# 4x RTX2080Ti @ 11 GB each — leave ~1 GB headroom per GPU for CUDA overhead
MAX_MEMORY = {
    0: "10GiB",
    1: "10GiB",
    2: "10GiB",
    3: "10GiB",
    "cpu": "40GiB",
}

# ── Shared judge system prompt ─────────────────────────────────────────────────
SYSTEM_CONTENT = """\
You are an expert evaluator for English-to-Telugu translations.
Given the original English text, its Telugu translation variants, and the sentiment label,
rate the overall translation quality on a scale of 0-5:
  0 = completely wrong or incomprehensible
  1 = very poor – major semantic or grammatical errors
  2 = poor – several noticeable errors
  3 = acceptable – minor errors, meaning mostly preserved
  4 = good – mostly accurate and natural
  5 = excellent – highly accurate and fluent

Respond ONLY in this exact JSON format with no additional text:
{"score": <integer 0-5>, "corrections": "<concise corrections, or 'None' if perfect>"}"""


# ── Prompt builders ────────────────────────────────────────────────────────────
def _user_content(example: dict) -> str:
    utterences_str = "\n".join(
        f"  Variant {i + 1}: {u}" for i, u in enumerate(example["utterence"])
    )
    return (
        f"Original English:\n  {example['input_text']}\n\n"
        f"Telugu translations:\n{utterences_str}\n\n"
        f"Sentiment label: {example['label']}\n\n"
        "Evaluate the translations and respond only in JSON."
    )


def build_messages(example: dict) -> list:
    return [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": _user_content(example)},
    ]


def build_completion_prompt(example: dict) -> str:
    """Prompt for pure-completion models that lack instruct fine-tuning (e.g. bloom-7b1)."""
    return (
        f"{SYSTEM_CONTENT}\n\n"
        "### Evaluation request\n"
        f"{_user_content(example)}\n\n"
        '### Response\n{"score": '
    )

def _falcon_prompt(example: dict) -> str:
    """Falcon-7b-instruct uses a simple User/Falcon turn format (no chat template)."""
    user_msg = f"{SYSTEM_CONTENT}\n\n{_user_content(example)}"
    return f"User: {user_msg}\n\nFalcon:"

# ── Response parser ────────────────────────────────────────────────────────────
def parse_response(text: str, prepend: str = "") -> dict:
    """Extract score and corrections from raw model output."""
    full = (prepend + text) if prepend else text

    # Attempt structured JSON extraction
    try:
        start = full.index("{")
        depth, end = 0, len(full)
        for idx in range(start, len(full)):
            if full[idx] == "{":
                depth += 1
            elif full[idx] == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        data = json.loads(full[start:end])
        score = data.get("score", -1)
        corrections = data.get("corrections", "")
        if isinstance(score, (int, float)) and 0 <= int(score) <= 5:
            return {"score": int(score), "corrections": str(corrections)}
    except (ValueError, json.JSONDecodeError):
        pass

    # Regex fallback
    sm = re.search(r'"score"\s*:\s*([0-5])', full)
    cm = re.search(r'"corrections"\s*:\s*"([^"]*)"', full, re.DOTALL)
    return {
        "score": int(sm.group(1)) if sm else -1,
        "corrections": cm.group(1).strip() if cm else full.strip()[:500],
    }


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model(model_id: str, retries: int = 3):
    print(f"\n[INFO] Loading {model_id} …")
    is_large = any(x in model_id for x in ["13b", "13B"])
    # Falcon is fully ported into transformers; bypassing legacy remote code
    # fixes its 'NoneType has no attribute shape' generate bug.
    use_remote_code = "falcon" not in model_id.lower()

    load_kwargs = {
        "trust_remote_code": use_remote_code,
        "device_map": "auto",
        "max_memory": MAX_MEMORY,
    }
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    if is_large:
        # Prefer 4-bit quantisation (~7 GB) but fall back to fp16 sharded across
        # all 4 GPUs (4×10 GiB = 40 GiB) when bitsandbytes is not available.
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"[INFO] Using 4-bit quantisation for {model_id}")
        except Exception as bnb_exc:
            print(
                f"[WARN] bitsandbytes unavailable ({bnb_exc}); "
                "loading in fp16 — device_map will shard across GPUs."
            )
            load_kwargs["dtype"] = torch.float16
    else:
        # 'dtype' is the non-deprecated replacement for 'torch_dtype'
        load_kwargs["dtype"] = torch.float16

    tok_kwargs = {"trust_remote_code": use_remote_code}
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.eval()
            print(f"[INFO] {model_id} ready.")
            return model, tokenizer
        except Exception as exc:
            last_exc = exc
            # Don't retry gated-repo auth errors — they won't resolve on retry
            if "403" in str(exc) or "gated" in str(exc).lower():
                break
            if attempt < retries:
                wait = 15 * attempt
                print(f"[WARN] Load attempt {attempt} failed: {exc}")
                print(f"       Retrying in {wait}s…")
                time.sleep(wait)
    raise last_exc


# ── Single inference pass ──────────────────────────────────────────────────────
def run_single(model, tokenizer, model_id: str, example: dict, run_idx: int) -> dict:
    is_completion = "bloom" in model_id.lower()
    is_falcon = "falcon" in model_id.lower()
    prepend_for_parse = ""

    if is_completion:
        raw_prompt = build_completion_prompt(example)
        prepend_for_parse = '{"score": '
        inputs = tokenizer(
            raw_prompt, return_tensors="pt", truncation=True, max_length=1536
        )
    elif is_falcon:
        # Falcon uses a simple User/Falcon turn format — no chat template
        raw_prompt = _falcon_prompt(example)
        inputs = tokenizer(
            raw_prompt, return_tensors="pt", truncation=True, max_length=2048
        )
    else:
        messages = build_messages(example)
        raw_prompt = _apply_chat_template(tokenizer, messages)
        inputs = tokenizer(
            raw_prompt, return_tensors="pt", truncation=True, max_length=2048
        )

    # Capture length BEFORE generate so legacy model code cannot clobber it
    # (fixes Falcon AttributeError: 'NoneType' object has no attribute 'shape')
    input_length = inputs["input_ids"].shape[1]

    # Move inputs to the device holding the first model parameter
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed = parse_response(response_text, prepend=prepend_for_parse)
    return {
        "input": example["input_text"],
        "utterence": example["utterence"],
        "label": example["label"],
        "score": parsed["score"],
        "corrections": parsed["corrections"],
        "raw_response": response_text,
        "run_idx": run_idx,
    }


def _apply_chat_template(tokenizer, messages: list) -> str:
    """Apply chat template with graceful fallbacks for models that lack system-role support."""
    # Attempt 1: standard chat template with system role
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        pass

    # Attempt 2: merge system content into user message (for Gemma-style templates)
    merged_content = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        pass

    # Attempt 3: generic Alpaca-style format
    return (
        f"### System:\n{messages[0]['content']}\n\n"
        f"### User:\n{messages[1]['content']}\n\n"
        "### Assistant:\n"
    )


# ── Per-model processing ───────────────────────────────────────────────────────
def process_model(model_id: str, example: dict) -> None:
    model_name = model_id.split("/")[-1]
    out_path = os.path.join(OUTPUT_DIR, f"100_judgements_{model_name}.json")

    # Resume support: load any partial results
    results: list = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"[INFO] Resuming {model_id}: {len(results)}/{NUM_RUNS} runs done.")
        except Exception:
            results = []

    start = len(results)
    if start >= NUM_RUNS:
        print(f"[INFO] {model_id}: already complete, skipping.")
        return

    try:
        model, tokenizer = load_model(model_id)
    except Exception as exc:
        warnings.warn(f"[ERROR] Could not load {model_id}: {exc}")
        return

    for i in range(start, NUM_RUNS):
        print(f"  [{model_name}] Run {i + 1:3d}/{NUM_RUNS} … ", end="", flush=True)
        try:
            result = run_single(model, tokenizer, model_id, example, i)
            print(f"score={result['score']}")
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                isinstance(exc, RuntimeError)
                and "out of memory" in str(exc).lower()
            )
            tag = "OOM" if oom else type(exc).__name__
            warnings.warn(f"[{tag}] Run {i + 1} for {model_id}: {exc}")
            if oom:
                torch.cuda.empty_cache()
            result = {
                "input": example["input_text"],
                "utterence": example["utterence"],
                "label": example["label"],
                "score": -1,
                "corrections": f"{tag}: {str(exc)[:300]}",
                "raw_response": "",
                "run_idx": i,
                "error": tag,
            }
            print(f"ERROR ({tag})")

        results.append(result)
        # Persist after every run so partial progress is never lost
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] {model_id} done. → {out_path}")

    # Release GPU memory before loading the next model
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    example = data[EXAMPLE_IDX]
    print(f"[INFO] Input      : {example['input_text'][:80]}…")
    print(f"[INFO] Label      : {example['label']}")
    print(f"[INFO] Variants   : {len(example['utterence'])}")
    print(f"[INFO] Runs/model : {NUM_RUNS}")
    print(f"[INFO] Output dir : {OUTPUT_DIR}\n")

    for model_id in MODELS:
        if model_id in COMPLETED_MODELS:
            print(f"[SKIP] {model_id}: already completed 100/100 runs.")
            continue
        process_model(model_id, example)

    print("\n[DONE] All models processed.")


if __name__ == "__main__":
    main()
