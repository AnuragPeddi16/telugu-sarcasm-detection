#!/usr/bin/env python3
"""Experiment 3: Refined Output via Tencent Chimera Model.

For each sample, pass 7 model translations (excluding sarvamai and tencent)
to tencent/HY-MT1.5-1.8B with a refinement prompt. Run 3 times per sample
with shuffled translation order. Save results with checkpoints.
"""

import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ───────────────────────────────────────────────────────────
MODEL_ID = "tencent/Hunyuan-MT-Chimera-7B"
INPUT_FILE = Path("Outputs/All Models Test/merged_translations.json")
OUTPUT_DIR = Path("/ssd_scratch/phani_ssd/model_out")
OUTPUT_FILE = OUTPUT_DIR / "refined_translations_chimera.json"
CHECKPOINT_EVERY = 100
NUM_RUNS = 3
MAX_NEW_TOKENS = 512

EXCLUDE_MODELS = {"sarvamai_sarvam-translate", "tencent_HY-MT1.5-1.8B"}

# Exact Chimera prompt template from the model card.
PROMPT_TEMPLATE = (
    "Analyze the following multiple {target_language} translations of the "
    "{source_language} segment surrounded in triple backticks and generate a "
    "single refined {target_language} translation. Only output the refined "
    "translation, do not explain.\n\n"
    "The {source_language} segment:\n"
    "```{source_text}```\n\n"
    "The multiple {target_language} translations:\n"
    "{translations}"
)
SOURCE_LANGUAGE = "English"
TARGET_LANGUAGE = "Telugu"

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(source_text: str, translations: List[str]) -> str:
    numbered = "\n".join(
        f"{i+1}. ```{t}```" for i, t in enumerate(translations)
    )
    return PROMPT_TEMPLATE.format(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        source_text=source_text,
        translations=numbered,
    )


def load_model_and_tokenizer():
    """Load Hunyuan-MT-Chimera-7B (BF16, ~16 GB) across 4 × RTX 2080 Ti (10 GB each)."""
    print(f"Loading {MODEL_ID} across 4 GPUs …")

    max_memory = {
        0: "9GiB",
        1: "9GiB",
        2: "9GiB",
        3: "9GiB",
        "cpu": "24GiB",
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Device map: {getattr(model, 'hf_device_map', 'N/A')}")
    return model, tokenizer


def generate_single(model, tokenizer, prompt: str) -> str:
    """Run a single generation pass using generation params from the model card."""
    # Build chat message and apply the model's chat template.
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Move to first device in the model's device map.
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.6,
            top_k=20,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    new_tokens = generated[0][prompt_len:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False).strip()
    return output


def get_translations_for_sample(sample: Dict[str, Any]) -> List[str]:
    """Extract translations from included models."""
    utterences = sample.get("utterence", {})
    return [
        utterences[model]
        for model in sorted(utterences.keys())
        if model not in EXCLUDE_MODELS and utterences[model] is not None
    ]


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    dataset = load_json(INPUT_FILE)
    print(f"Loaded {len(dataset)} samples")

    model, tokenizer = load_model_and_tokenizer()

    # Resume from checkpoint if it exists
    checkpoint_file = OUTPUT_DIR / "refined_chimera_checkpoint.json"
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if checkpoint_file.exists():
        results = load_json(checkpoint_file)
        start_idx = len(results)
        print(f"Resuming from checkpoint at index {start_idx}")

    for idx in tqdm(range(start_idx, len(dataset)), desc="Refining", unit="sample"):
        sample = dataset[idx]
        source_text = sample["input"]
        label = sample.get("label")
        translations = get_translations_for_sample(sample)

        if not translations:
            results.append({
                "input_text": source_text,
                "utterence": [],
                "label": label,
            })
            continue

        refined_outputs: List[str] = []
        for run in tqdm(range(NUM_RUNS), desc=f"  Runs [sample {idx+1}]", leave=False):
            # Shuffle translation order for each run
            shuffled = translations.copy()
            random.shuffle(shuffled)
            prompt = build_prompt(source_text, shuffled)
            try:
                output = generate_single(model, tokenizer, prompt)
            except torch.cuda.OutOfMemoryError as e:
                output = f"ERROR: CUDA out of memory — {e}"
                print(f"\nWARNING: CUDA OOM at sample {idx}, run {run}: {e}", flush=True)
                torch.cuda.empty_cache()
            except Exception as e:
                output = f"ERROR: {type(e).__name__} — {e}"
                print(f"\nWARNING: Error at sample {idx}, run {run}: {type(e).__name__}: {e}", flush=True)
            refined_outputs.append(output)

        results.append({
            "input_text": source_text,
            "utterence": refined_outputs,
            "label": label,
        })

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_json(checkpoint_file, results)
            print(f"\n  Checkpoint saved at index {idx + 1}")

    # Final save
    save_json(OUTPUT_FILE, results)
    print(f"\nDone! {len(results)} samples saved to {OUTPUT_FILE}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
