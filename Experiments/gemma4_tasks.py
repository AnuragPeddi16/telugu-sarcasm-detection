#!/usr/bin/env python3
"""Gemma 4 Model Evaluation Script.

Tests all Gemma 4 variants on:
  1. Translation (English → Telugu, preserving sarcasm)
  2. Judge (evaluate refined_translations_chimera.json with scores + corrections)
  3. Refiner (use judge corrections to fix translations)

Both thinking and non-thinking modes. Outputs saved to /ssd_scratch/phani_ssd/model_out.

Hardware: 4x RTX 2080 Ti (10GB each), 60GB CPU RAM

REQUIREMENTS:
  pip install "transformers>=5.5.4" accelerate torch tqdm
  (transformers 4.x does NOT support model_type="gemma4" — AutoProcessor raises
   "Unrecognized processing class" on every Gemma 4 model)
"""

import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Prevent CUDA OOM during model loading due to memory fragmentation after
# unloading a previous model. Without this, PyTorch may fail to allocate a
# large contiguous block even when enough total free memory exists.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Transformers version guard ────────────────────────────────────────────────
# Gemma 4 (model_type="gemma4", Gemma4Processor) requires transformers >= 5.x.
# Version 4.x only knows gemma / gemma2 / gemma3 / gemma3n — AutoProcessor will
# raise "Unrecognized processing class" for every Gemma 4 model.
# Fix: pip install "transformers>=5.5.4" accelerate torch
import transformers as _tfm
_tfm_ver = tuple(int(x) for x in _tfm.__version__.split(".")[:2])
if _tfm_ver < (5, 0):
    print(
        f"ERROR: transformers {_tfm.__version__} does not support Gemma 4.\n"
        "Gemma 4 (model_type='gemma4') requires transformers >= 5.x.\n"
        'Run: pip install "transformers>=5.5.4" accelerate torch\n'
        "Then re-run this script.",
        file=sys.stderr,
    )
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("gemma4/model_out")
DATASET_FILE = Path("gemma4/5000-samples.json")
JUDGE_INPUT_FILE = Path("gemma4/refined_translations_chimera.json")
CHECKPOINT_EVERY = 50

# ── Run Control ───────────────────────────────────────────────────────────────
# Set each task to True/False to enable or skip it.
TASKS_TO_RUN = {
    "translation": True,
    "judge": False,
    "refiner": False,
}

# Restrict which models to run by listing their short_names.
# Set to None to run ALL models defined in MODELS.
# e.g., MODELS_TO_RUN = ["gemma-4-E2B-it", "gemma-4-E4B-it"]
MODELS_TO_RUN = None  # None = all models

# For test runs, switch these to the Test directory:
# DATASET_FILE = Path("gemma4/Test/5000-samples.json")
# JUDGE_INPUT_FILE = Path("gemma4/Test/refined_translations_chimera.json")

# Gemma 4 recommended sampling parameters
SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "do_sample": True,
}

# Model configurations with memory planning for 4x RTX 2080 Ti (10GB each) + 60GB CPU
MODELS = [
    {
        "model_id": "google/gemma-4-E2B-it",
        "short_name": "gemma-4-E2B-it",
        "total_params_gb": 10,  # 5.1B with embeddings in BF16
        "quantize_4bit": False,
        "torch_dtype": "bfloat16",
        "max_memory": {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "30GiB"},
    },
    {
        "model_id": "google/gemma-4-E4B-it",
        "short_name": "gemma-4-E4B-it",
        "total_params_gb": 16,  # 8B with embeddings in BF16
        "quantize_4bit": False,
        "torch_dtype": "bfloat16",
        "max_memory": {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "35GiB"},
    },
    {
        "model_id": "google/gemma-4-26B-A4B-it",
        "short_name": "gemma-4-26B-A4B-it",
        "total_params_gb": 50,  # 25.2B MoE in BF16 ~50GB, overflow to CPU
        "quantize_4bit": False,
        "torch_dtype": "bfloat16",
        "max_memory": {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "50GiB"},
    },
    {
        "model_id": "google/gemma-4-31B-it",
        "short_name": "gemma-4-31B-it",
        "total_params_gb": 62,  # 30.7B dense in BF16 ~62GB, overflow to CPU
        "quantize_4bit": False,
        "torch_dtype": "bfloat16",
        "max_memory": {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "55GiB"},
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert English-to-Telugu translator. Your task is to translate "
    "English text into Telugu while preserving the tone, intent, and stylistic "
    "elements including sarcasm, irony, and humor. Do not add explanations."
)

TRANSLATION_USER_PROMPT = (
    "Translate the following English text to Telugu. Preserve the sarcastic tone "
    "and meaning accurately.\n\n"
    "English: {text}\n\n"
    "Telugu translation:"
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert bilingual evaluator for English-to-Telugu translations. "
    "You evaluate translation quality focusing on accuracy, fluency, and "
    "preservation of tone (especially sarcasm). Respond ONLY in valid JSON."
)

JUDGE_USER_PROMPT = (
    "Evaluate the following Telugu translations of the given English text.\n\n"
    "English (Original): {input_text}\n\n"
    "Telugu Translations:\n{translations}\n\n"
    "Rate the overall quality on a scale of 1-5:\n"
    "1 = Very poor (meaning lost, sarcasm destroyed)\n"
    "2 = Poor (major errors, sarcasm weakened)\n"
    "3 = Acceptable (minor errors, sarcasm partially preserved)\n"
    "4 = Good (accurate, sarcasm well preserved)\n"
    "5 = Excellent (perfect translation, sarcasm fully preserved)\n\n"
    "Provide corrections if the score is less than 5. If no corrections needed, "
    'write "None".\n\n'
    "Respond in this exact JSON format:\n"
    '{{"score": <1-5>, "corrections": "<corrections or None>"}}'
)

REFINER_SYSTEM_PROMPT = (
    "You are an expert English-to-Telugu translator specializing in refining "
    "translations based on feedback. Apply the given corrections to improve "
    "the translations. Output only the corrected Telugu translations."
)

REFINER_USER_PROMPT = (
    "Original English text: {input_text}\n\n"
    "Current Telugu translations:\n{translations}\n\n"
    "Corrections/Feedback: {corrections}\n\n"
    "Provide the corrected Telugu translations based on the feedback above. "
    "Output each corrected translation on a new line, numbered 1, 2, 3."
)


# ══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_processor(model_config: Dict[str, Any]):
    """Load a Gemma 4 model with appropriate memory config."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = model_config["model_id"]
    print(f"\n{'='*70}")
    print(f"Loading model: {model_id}")
    print(f"Quantize 4-bit: {model_config['quantize_4bit']}")
    print(f"Max memory: {model_config['max_memory']}")
    print(f"{'='*70}\n")

    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs = {
        "device_map": "auto",
        "max_memory": model_config["max_memory"],
        "low_cpu_mem_usage": True,
        # RTX 2080 Ti is Turing — FlashAttention requires Ampere+. Force eager.
        "attn_implementation": "eager",
    }

    if model_config["quantize_4bit"]:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    print(f"Model loaded successfully. Device map: {getattr(model, 'hf_device_map', 'N/A')}")
    return model, processor


def unload_model():
    """Free GPU/CPU memory after the caller has deleted model/processor refs."""
    free_memory()
    print("Model unloaded and memory freed.")


# ══════════════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    model,
    processor,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
    max_new_tokens: int,
) -> str:
    """Generate a response from the model."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # With device_map="auto" + CPU offload, model.device may be "cpu".
    # Always prefer the first CUDA device for inputs; accelerate handles the rest.
    input_device = "cuda:0" if torch.cuda.is_available() else model.device
    inputs = processor(text=text, return_tensors="pt").to(input_device)
    input_len = inputs["input_ids"].shape[-1]

    # Resolve pad_token_id safely (tokenizer is always available via ProcessorMixin)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **SAMPLING_PARAMS,
            pad_token_id=pad_token_id,
        )

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response)

    # parse_response returns a dict with 'content' (and optionally 'thought')
    if isinstance(parsed, dict):
        return parsed.get("content", str(parsed))
    return str(parsed)


# ══════════════════════════════════════════════════════════════════════════════
# Task 1: Translation
# ══════════════════════════════════════════════════════════════════════════════

def run_translation(
    model,
    processor,
    dataset: List[Dict[str, Any]],
    enable_thinking: bool,
    output_path: Path,
    checkpoint_path: Path,
) -> List[Dict[str, Any]]:
    """Translate all samples from English to Telugu preserving sarcasm."""
    max_new_tokens = 1024 if enable_thinking else 512
    mode_str = "thinking" if enable_thinking else "non-thinking"
    print(f"\n--- Translation Task ({mode_str}) ---")
    print(f"Samples: {len(dataset)}, Output: {output_path}")

    # Skip if already complete (final output exists and no pending checkpoint)
    if output_path.exists() and not checkpoint_path.exists():
        existing = load_json(output_path)
        print(f"Skipping: completed output already exists ({len(existing)} samples) at {output_path}")
        return existing

    # Resume from checkpoint
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if checkpoint_path.exists():
        results = load_json(checkpoint_path)
        start_idx = len(results)
        print(f"Resuming from checkpoint at index {start_idx}")

    for idx in tqdm(range(start_idx, len(dataset)), desc=f"Translation ({mode_str})"):
        sample = dataset[idx]
        # The dataset has "utterance" as the English text field
        input_text = sample.get("utterance", sample.get("utterence", ""))
        label = sample.get("label", "")

        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": TRANSLATION_USER_PROMPT.format(text=input_text)},
        ]

        try:
            translation = generate_response(
                model, processor, messages, enable_thinking, max_new_tokens
            )
        except torch.OutOfMemoryError as e:
            print(f"\n[WARN] OOM at index {idx}: {e}")
            free_memory()
            translation = f"OOM_ERROR: {str(e)[:200]}"
        except Exception as e:
            print(f"\n[WARN] Error at index {idx}: {e}")
            translation = f"ERROR: {str(e)[:200]}"

        results.append({
            "input": input_text,
            "utterence": translation,
            "label": label,
        })

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_json(checkpoint_path, results)

    # Save final output
    save_json(output_path, results)
    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Translation complete. Saved {len(results)} results to {output_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Task 2: Judge
# ══════════════════════════════════════════════════════════════════════════════

def run_judge(
    model,
    processor,
    judge_data: List[Dict[str, Any]],
    enable_thinking: bool,
    output_path: Path,
    checkpoint_path: Path,
) -> List[Dict[str, Any]]:
    """Evaluate each translation set with scores and corrections."""
    max_new_tokens = 2048 if enable_thinking else 1024
    mode_str = "thinking" if enable_thinking else "non-thinking"
    print(f"\n--- Judge Task ({mode_str}) ---")
    print(f"Samples: {len(judge_data)}, Output: {output_path}")

    if output_path.exists() and not checkpoint_path.exists():
        existing = load_json(output_path)
        print(f"Skipping: completed output already exists ({len(existing)} samples) at {output_path}")
        return existing

    # Resume from checkpoint
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if checkpoint_path.exists():
        results = load_json(checkpoint_path)
        start_idx = len(results)
        print(f"Resuming from checkpoint at index {start_idx}")

    for idx in tqdm(range(start_idx, len(judge_data)), desc=f"Judge ({mode_str})"):
        sample = judge_data[idx]
        input_text = sample.get("input_text", sample.get("input", ""))
        utterences = sample.get("utterence", [])
        label = sample.get("label", "")

        # Format translations for the prompt
        if isinstance(utterences, list):
            translations_str = "\n".join(
                f"{i+1}. {t}" for i, t in enumerate(utterences)
            )
        else:
            translations_str = str(utterences)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_PROMPT.format(
                input_text=input_text,
                translations=translations_str,
            )},
        ]

        try:
            raw_response = generate_response(
                model, processor, messages, enable_thinking, max_new_tokens
            )
            # Parse JSON from response
            score, corrections = parse_judge_response(raw_response)
        except torch.OutOfMemoryError as e:
            print(f"\n[WARN] OOM at index {idx}: {e}")
            free_memory()
            raw_response = f"OOM_ERROR: {str(e)[:200]}"
            score = -1
            corrections = f"OOM_ERROR: {str(e)[:200]}"
        except Exception as e:
            print(f"\n[WARN] Error at index {idx}: {e}")
            raw_response = f"ERROR: {str(e)[:200]}"
            score = -1
            corrections = f"ERROR: {str(e)[:200]}"

        results.append({
            "input": input_text,
            "utterence": utterences,
            "label": label,
            "score": score,
            "corrections": corrections,
            "raw_response": raw_response,
        })

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_json(checkpoint_path, results)

    # Save final output
    save_json(output_path, results)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Judge complete. Saved {len(results)} results to {output_path}")
    return results


def parse_judge_response(response: str) -> tuple:
    """Extract score and corrections from judge JSON response."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = int(data.get("score", -1))
            corrections = data.get("corrections", "None")
            return score, corrections
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract score from text
    score_match = re.search(r'"?score"?\s*[:=]\s*(\d)', response)
    score = int(score_match.group(1)) if score_match else -1
    corrections_match = re.search(
        r'"?corrections"?\s*[:=]\s*"?(.+?)(?:"|$|\})', response, re.DOTALL
    )
    corrections = corrections_match.group(1).strip() if corrections_match else response
    return score, corrections


# ══════════════════════════════════════════════════════════════════════════════
# Task 3: Refiner
# ══════════════════════════════════════════════════════════════════════════════

def run_refiner(
    model,
    processor,
    judge_results: List[Dict[str, Any]],
    enable_thinking: bool,
    output_path: Path,
    checkpoint_path: Path,
) -> List[Dict[str, Any]]:
    """Use judge corrections to refine translations."""
    max_new_tokens = 1024 if enable_thinking else 512
    mode_str = "thinking" if enable_thinking else "non-thinking"
    print(f"\n--- Refiner Task ({mode_str}) ---")
    print(f"Samples: {len(judge_results)}, Output: {output_path}")

    if output_path.exists() and not checkpoint_path.exists():
        existing = load_json(output_path)
        print(f"Skipping: completed output already exists ({len(existing)} samples) at {output_path}")
        return existing

    # Resume from checkpoint
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if checkpoint_path.exists():
        results = load_json(checkpoint_path)
        start_idx = len(results)
        print(f"Resuming from checkpoint at index {start_idx}")

    for idx in tqdm(range(start_idx, len(judge_results)), desc=f"Refiner ({mode_str})"):
        sample = judge_results[idx]
        input_text = sample.get("input", "")
        utterences = sample.get("utterence", [])
        label = sample.get("label", "")
        corrections = sample.get("corrections", "None")

        # Skip if no corrections needed or score is perfect
        if corrections in ("None", "none", "") or sample.get("score", 0) == 5:
            results.append({
                "input": input_text,
                "utterence": utterences,
                "label": label,
                "refined": False,
                "corrections_applied": "None",
            })
            continue

        # Format translations
        if isinstance(utterences, list):
            translations_str = "\n".join(
                f"{i+1}. {t}" for i, t in enumerate(utterences)
            )
        else:
            translations_str = str(utterences)

        messages = [
            {"role": "system", "content": REFINER_SYSTEM_PROMPT},
            {"role": "user", "content": REFINER_USER_PROMPT.format(
                input_text=input_text,
                translations=translations_str,
                corrections=corrections,
            )},
        ]

        try:
            refined_output = generate_response(
                model, processor, messages, enable_thinking, max_new_tokens
            )
        except torch.OutOfMemoryError as e:
            print(f"\n[WARN] OOM at index {idx}: {e}")
            free_memory()
            refined_output = f"OOM_ERROR: {str(e)[:200]}"
        except Exception as e:
            print(f"\n[WARN] Error at index {idx}: {e}")
            refined_output = f"ERROR: {str(e)[:200]}"

        results.append({
            "input": input_text,
            "utterence": refined_output,
            "label": label,
            "refined": True,
            "corrections_applied": corrections,
        })

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_json(checkpoint_path, results)

    # Save final output
    save_json(output_path, results)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Refiner complete. Saved {len(results)} results to {output_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tasks_for_model(
    model_config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    judge_data: List[Dict[str, Any]],
):
    """Load a model once and run all tasks (both thinking modes) before unloading."""
    short_name = model_config["short_name"]

    # Decide work upfront so we can avoid loading the model if everything is done.
    thinking_modes = [False, True]
    pending = []
    for enable_thinking in thinking_modes:
        mode_suffix = "_thinking" if enable_thinking else ""
        translation_output = OUTPUT_DIR / f"Translation{mode_suffix}_{short_name}.json"
        translation_checkpoint = OUTPUT_DIR / f"checkpoint_Translation{mode_suffix}_{short_name}.json"
        judge_output = OUTPUT_DIR / f"judge{mode_suffix}_{short_name}.json"
        judge_checkpoint = OUTPUT_DIR / f"checkpoint_judge{mode_suffix}_{short_name}.json"
        refiner_output = OUTPUT_DIR / f"refiner{mode_suffix}_{short_name}.json"
        refiner_checkpoint = OUTPUT_DIR / f"checkpoint_refiner{mode_suffix}_{short_name}.json"

        translation_done = (
            not TASKS_TO_RUN.get("translation", True)
            or (translation_output.exists() and not translation_checkpoint.exists())
        )
        judge_done = (
            not TASKS_TO_RUN.get("judge", True)
            or (judge_output.exists() and not judge_checkpoint.exists())
        )
        refiner_done = (
            not TASKS_TO_RUN.get("refiner", True)
            or (refiner_output.exists() and not refiner_checkpoint.exists())
        )
        if translation_done and judge_done and refiner_done:
            print(f"\nSkipping {short_name} ({'thinking' if enable_thinking else 'non-thinking'}): all enabled tasks already complete")
            continue
        pending.append(enable_thinking)

    if not pending:
        print(f"\nAll enabled tasks already complete for {short_name}; not loading model.")
        return

    model, processor = load_model_and_processor(model_config)
    for enable_thinking in pending:
        mode_suffix = "_thinking" if enable_thinking else ""
        print(f"\n{'#'*70}")
        print(f"# Mode: {'THINKING' if enable_thinking else 'NON-THINKING'} | Model: {short_name}")
        print(f"{'#'*70}")

        # ── Task 1: Translation ──────────────────────────────────────────
        translation_output = OUTPUT_DIR / f"Translation{mode_suffix}_{short_name}.json"
        translation_checkpoint = OUTPUT_DIR / f"checkpoint_Translation{mode_suffix}_{short_name}.json"

        if TASKS_TO_RUN.get("translation", True):
            run_translation(
                model, processor, dataset,
                enable_thinking=enable_thinking,
                output_path=translation_output,
                checkpoint_path=translation_checkpoint,
            )
            free_memory()
        else:
            print("Skipping translation (disabled in TASKS_TO_RUN)")

        # ── Task 2: Judge ────────────────────────────────────────────────
        judge_output = OUTPUT_DIR / f"judge{mode_suffix}_{short_name}.json"
        judge_checkpoint = OUTPUT_DIR / f"checkpoint_judge{mode_suffix}_{short_name}.json"

        judge_results: Optional[List[Dict[str, Any]]] = None
        if TASKS_TO_RUN.get("judge", True):
            judge_results = run_judge(
                model, processor, judge_data,
                enable_thinking=enable_thinking,
                output_path=judge_output,
                checkpoint_path=judge_checkpoint,
            )
            free_memory()
        else:
            print("Skipping judge (disabled in TASKS_TO_RUN)")
            # Load existing judge output so refiner can still run if needed
            if TASKS_TO_RUN.get("refiner", True) and judge_output.exists():
                judge_results = load_json(judge_output)

        # ── Task 3: Refiner ──────────────────────────────────────────────
        refiner_output = OUTPUT_DIR / f"refiner{mode_suffix}_{short_name}.json"
        refiner_checkpoint = OUTPUT_DIR / f"checkpoint_refiner{mode_suffix}_{short_name}.json"

        if TASKS_TO_RUN.get("refiner", True):
            if judge_results is None:
                print("Skipping refiner: no judge results available (run judge first)")
            else:
                run_refiner(
                    model, processor, judge_results,
                    enable_thinking=enable_thinking,
                    output_path=refiner_output,
                    checkpoint_path=refiner_checkpoint,
                )
                free_memory()
        else:
            print("Skipping refiner (disabled in TASKS_TO_RUN)")

    # Unload model: delete ALL references here so refcount drops to 0,
    # THEN call free_memory(). Deleting inside unload_model() only removes
    # the local copy — the caller's variable keeps the tensors alive.
    del model, processor
    unload_model()


def main():
    """Main entry point: run all models sequentially."""
    print("=" * 70)
    print("Gemma 4 Evaluation Pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Judge input: {JUDGE_INPUT_FILE}")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    dataset = load_json(DATASET_FILE) if TASKS_TO_RUN.get("translation", True) else []
    judge_data = (
        load_json(JUDGE_INPUT_FILE)
        if TASKS_TO_RUN.get("judge", True) or TASKS_TO_RUN.get("refiner", True)
        else []
    )
    if dataset:
        print(f"  Translation dataset: {len(dataset)} samples")
    if judge_data:
        print(f"  Judge dataset: {len(judge_data)} samples")

    # Filter models based on MODELS_TO_RUN
    active_models = (
        MODELS
        if MODELS_TO_RUN is None
        else [m for m in MODELS if m["short_name"] in MODELS_TO_RUN]
    )
    print(f"  Active tasks: {[k for k, v in TASKS_TO_RUN.items() if v]}")
    print(f"  Active models: {[m['short_name'] for m in active_models]}")

    # Process each model
    for model_config in active_models:
        print(f"\n\n{'*'*70}")
        print(f"* Starting model: {model_config['model_id']}")
        print(f"{'*'*70}")

        try:
            run_all_tasks_for_model(model_config, dataset, judge_data)
        except Exception as e:
            print(f"\n!!! FATAL ERROR with {model_config['model_id']}: {e}")
            import traceback
            traceback.print_exc()
            free_memory()
            continue

    print("\n" + "=" * 70)
    print("All models processed. Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
