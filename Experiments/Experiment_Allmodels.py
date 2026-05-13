import gc
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    SeamlessM4Tv2ForTextToText,
)

try:
    from IndicTransToolkit.processor import IndicProcessor
except Exception:
    IndicProcessor = None


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_PROMPT_TEMPLATE = (
    "Translate the following text from English to Telugu.\n"
    "English: {text}\n"
    "Telugu:"
)

# Global experiment configuration (edit these values directly, no CLI needed).
DATASET_FILE: Path = Path("NLLB-200/5000-samples.json")
OUTPUT_DIR: Path = Path("/ssd_scratch/phani_ssd/model_out")
CHECKPOINT_EVERY: int = 100
MAX_GPUS: int = 4
GPU_RESERVE_GB: float = 0.5
CPU_TOTAL_RAM_GB: int = 72
CPU_RESERVE_GB: int = 4
CPU_MAX_MEMORY_GIB: int = max(CPU_TOTAL_RAM_GB - CPU_RESERVE_GB, 1)
INCLUDE_NIUTRANS_MODELS: bool = False
ONLY_MODELS: Optional[List[str]] = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    params_b: float
    prompt_style: str = "plain"
    trust_remote_code: bool = False


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("anithasoma/nllb-finetuned-telugu", "seq2seq", 1.3),
    ModelSpec("NiuTrans/LMT-60-1.7B", "causal", 1.7, prompt_style="plain"),
    ModelSpec("facebook/seamless-m4t-v2-large", "seamless", 2.3),
    ModelSpec("NiuTrans/LMT-60-0.6B", "causal", 0.6, prompt_style="user_chat"),
    ModelSpec("aryaumesh/english-to-telugu", "mbart", 1.0),
    ModelSpec(
        "prajdabre/rotary-indictrans2-en-indic-1B",
        "indictrans",
        1.0,
        trust_remote_code=True,
    ),
    ModelSpec("facebook/nllb-200-1.3B", "nllb", 1.3),
    ModelSpec("sarvamai/sarvam-translate", "causal", 1.2, prompt_style="system_chat"),
    ModelSpec("tencent/HY-MT1.5-1.8B", "causal", 1.8, prompt_style="plain"),
    ModelSpec("archita091234/fine-tuned-translation", "seq2seq", 1.0),
    ModelSpec("gvaishnavi/bilingual-translation", "mbart", 1.0),
]


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_dataset(dataset_file: Path) -> List[Dict[str, Any]]:
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with dataset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list. Got: {type(data).__name__}")
    normalized: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"utterance": str(item), "label": ""})
    return normalized


def build_gpu_memory_plan() -> Tuple[List[int], Dict[Any, str]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return [], {"cpu": f"{CPU_MAX_MEMORY_GIB}GiB"}

    use_gpus = min(MAX_GPUS, torch.cuda.device_count())
    reserve_bytes = int(GPU_RESERVE_GB * (1024 ** 3))

    gpu_ids: List[int] = []
    max_memory: Dict[Any, str] = {"cpu": f"{CPU_MAX_MEMORY_GIB}GiB"}

    for gpu_id in range(use_gpus):
        total_bytes = torch.cuda.get_device_properties(gpu_id).total_memory
        usable_bytes = max(total_bytes - reserve_bytes, 1 * (1024 ** 3))
        usable_gib = max(int(usable_bytes // (1024 ** 3)), 1)
        max_memory[gpu_id] = f"{usable_gib}GiB"
        gpu_ids.append(gpu_id)

        try:
            fraction = min(max(usable_bytes / float(total_bytes), 0.1), 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)
        except Exception:
            pass

    return gpu_ids, max_memory


def device_from_map(model: torch.nn.Module) -> torch.device:
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        for location in hf_map.values():
            if isinstance(location, int):
                return torch.device(f"cuda:{location}")
            if isinstance(location, str) and location.startswith("cuda"):
                return torch.device(location)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def clean_model_inputs(batch: Dict[str, Any], drop_token_type_ids: bool = False) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in batch.items():
        if value is None:
            continue
        if drop_token_type_ids and key == "token_type_ids":
            continue
        cleaned[key] = value
    return cleaned


def build_prompt(spec: ModelSpec, text: str) -> str:
    return BASE_PROMPT_TEMPLATE.format(text=text)


def model_load_kwargs(
    dtype: torch.dtype,
    use_accelerate: bool,
    max_memory: Optional[Dict[Any, str]],
    offload_dir: Optional[Path],
    trust_remote_code: bool,
    forced_device_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if use_accelerate and torch.cuda.is_available():
        kwargs["device_map"] = forced_device_map if forced_device_map is not None else "auto"
        kwargs["max_memory"] = max_memory
        if offload_dir is not None:
            offload_dir.mkdir(parents=True, exist_ok=True)
            kwargs["offload_folder"] = str(offload_dir)
    return kwargs


def load_single_worker(
    spec: ModelSpec,
    dtype: torch.dtype,
    use_accelerate: bool,
    max_memory: Optional[Dict[Any, str]],
    device: Optional[torch.device],
    offload_dir: Optional[Path],
    forced_device_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    worker: Dict[str, Any] = {"spec": spec, "device": device}
    kwargs = model_load_kwargs(
        dtype=dtype,
        use_accelerate=use_accelerate,
        max_memory=max_memory,
        offload_dir=offload_dir,
        trust_remote_code=spec.trust_remote_code,
        forced_device_map=forced_device_map,
    )

    if spec.family in {"seq2seq", "nllb"}:
        worker["tokenizer"] = AutoTokenizer.from_pretrained(spec.model_id)
        worker["model"] = AutoModelForSeq2SeqLM.from_pretrained(spec.model_id, **kwargs)
    elif spec.family == "mbart":
        worker["tokenizer"] = MBart50TokenizerFast.from_pretrained(spec.model_id)
        worker["model"] = MBartForConditionalGeneration.from_pretrained(spec.model_id, **kwargs)
    elif spec.family == "seamless":
        worker["processor"] = AutoProcessor.from_pretrained(spec.model_id)
        worker["model"] = SeamlessM4Tv2ForTextToText.from_pretrained(spec.model_id, **kwargs)
    elif spec.family == "indictrans":
        if IndicProcessor is None:
            raise RuntimeError(
                "IndicTransToolkit is not installed; required for prajdabre/rotary-indictrans2-en-indic-1B"
            )
        worker["processor"] = IndicProcessor(inference=True)
        worker["tokenizer"] = AutoTokenizer.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code
        )
        worker["model"] = AutoModelForSeq2SeqLM.from_pretrained(spec.model_id, **kwargs)
    elif spec.family == "causal":
        worker["tokenizer"] = AutoTokenizer.from_pretrained(spec.model_id)
        worker["model"] = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")

    if not use_accelerate:
        target_device = device if device is not None else torch.device("cpu")
        worker["model"] = worker["model"].to(target_device)

    worker["model"].eval()
    tokenizer = worker.get("tokenizer")
    if tokenizer is not None and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return worker


def load_workers_for_model(
    spec: ModelSpec,
    gpu_ids: List[int],
    dtype: torch.dtype,
    max_memory: Dict[Any, str],
    offload_root: Path,
) -> List[Dict[str, Any]]:
    is_small = spec.params_b < 2.0
    workers: List[Dict[str, Any]] = []

    if is_small:
        if gpu_ids:
            target_gpu_ids = gpu_ids[:4]
            for gpu_id in target_gpu_ids:
                single_max_memory: Dict[Any, str] = {
                    "cpu": max_memory.get("cpu", "128GiB"),
                    gpu_id: max_memory[gpu_id],
                }
                workers.append(
                    load_single_worker(
                        spec=spec,
                        dtype=dtype,
                        use_accelerate=True,
                        max_memory=single_max_memory,
                        device=None,
                        offload_dir=None,
                        forced_device_map={"": gpu_id},
                    )
                )
        else:
            workers.append(
                load_single_worker(
                    spec=spec,
                    dtype=torch.float32,
                    use_accelerate=False,
                    max_memory=None,
                    device=torch.device("cpu"),
                    offload_dir=None,
                )
            )
    else:
        offload_dir = offload_root / safe_model_name(spec.model_id)
        workers.append(
            load_single_worker(
                spec=spec,
                dtype=dtype,
                use_accelerate=True,
                max_memory=max_memory,
                device=None,
                offload_dir=offload_dir,
            )
        )

    return workers


def translate_once(worker: Dict[str, Any], text: str) -> str:
    spec: ModelSpec = worker["spec"]
    model = worker["model"]

    if spec.family in {"seq2seq", "nllb", "mbart", "indictrans"}:
        tokenizer = worker["tokenizer"]
    else:
        tokenizer = worker.get("tokenizer")

    if spec.family == "seamless":
        processor = worker["processor"]
        device = worker["device"] or device_from_map(model)
        inputs = processor(text=text, src_lang="eng", return_tensors="pt")
        inputs = move_batch_to_device(inputs, device)
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                tgt_lang="tel",
                num_beams=5,
                max_new_tokens=256,
                early_stopping=True,
            )
        return processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

    if spec.family == "indictrans":
        processor = worker["processor"]
        device = worker["device"] or device_from_map(model)
        batch = processor.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="tel_Telu")
        tokens = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        tokens = clean_model_inputs(tokens, drop_token_type_ids=True)
        if "input_ids" not in tokens:
            raise RuntimeError("IndicTrans tokenizer did not return input_ids.")
        if "attention_mask" not in tokens:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            tokens["attention_mask"] = tokens["input_ids"].ne(pad_id).long()
        tokens = move_batch_to_device(tokens, device)
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                num_beams=10,
                length_penalty=1.5,
                repetition_penalty=2.0,
                num_return_sequences=1,
                max_new_tokens=1024,
                early_stopping=True,
                use_cache=False,
            )
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        postprocessed = processor.postprocess_batch(decoded, lang="tel_Telu")
        return postprocessed[0]

    if spec.family in {"seq2seq", "nllb", "mbart"}:
        device = worker["device"] or device_from_map(model)
        if spec.family == "nllb":
            tokenizer.src_lang = "eng_Latn"
            forced_bos = tokenizer.convert_tokens_to_ids("tel_Telu")
        elif spec.family == "mbart":
            try:
                tokenizer.src_lang = "en_XX"
            except Exception:
                pass
            forced_bos = None
            if hasattr(tokenizer, "lang_code_to_id"):
                forced_bos = (
                    tokenizer.lang_code_to_id.get("te_IN")
                    or tokenizer.lang_code_to_id.get("tel_Telu")
                )
        else:
            forced_bos = None

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = move_batch_to_device(inputs, device)

        gen_kwargs: Dict[str, Any] = {
            "num_beams": 5,
            "max_new_tokens": 256,
            "early_stopping": True,
        }
        if forced_bos is not None and forced_bos >= 0:
            gen_kwargs["forced_bos_token_id"] = forced_bos

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    if spec.family == "causal":
        prompt = build_prompt(spec, text)
        if spec.prompt_style == "user_chat":
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                rendered = prompt
        elif spec.prompt_style == "system_chat":
            messages = [
                {"role": "system", "content": "Translate the text to Telugu only."},
                {"role": "user", "content": prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                rendered = prompt
        else:
            rendered = prompt

        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=2048)
        # Tencent/HY-MT1.5-1.8B throws if token_type_ids are passed to generate.
        drop_token_type_ids = spec.model_id == "tencent/HY-MT1.5-1.8B"
        inputs = clean_model_inputs(inputs, drop_token_type_ids=drop_token_type_ids)
        model_device = worker["device"] or device_from_map(model)
        inputs = move_batch_to_device(inputs, model_device)

        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if spec.model_id == "sarvamai/sarvam-translate":
            gen_kwargs["temperature"] = 0.01
        if spec.model_id == "tencent/HY-MT1.5-1.8B":
            gen_kwargs["num_beams"] = 5
            gen_kwargs["no_repeat_ngram_size"] = 3

        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[0][prompt_len:]
        translation = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return translation if translation else tokenizer.decode(
            generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    raise ValueError(f"Unsupported model family: {spec.family}")


def translate_dataset_for_model(
    spec: ModelSpec,
    workers: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    translated_rows: List[Dict[str, Any]] = []
    if not workers:
        raise RuntimeError(f"No workers available for model {spec.model_id}")
    num_workers = len(workers)
    with tqdm(total=len(dataset), desc="Translations", unit="sample") as pbar:
        for idx, entry in enumerate(dataset, start=1):
            source_text = entry.get("utterance", "")
            source_text = source_text if isinstance(source_text, str) else str(source_text)
            label = entry.get("label", "")
            label = label if isinstance(label, str) else str(label)

            if source_text.strip():
                worker = workers[(idx - 1) % num_workers]
                try:
                    translated_text = translate_once(worker, source_text)
                except Exception as exc:
                    translated_text = f"[ERROR] {exc}"
            else:
                translated_text = source_text

            translated_rows.append(
                {
                    "input": source_text,
                    "utterence": translated_text,
                    "label": label,
                }
            )

            if CHECKPOINT_EVERY > 0 and idx % CHECKPOINT_EVERY == 0:
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(translated_rows, f, ensure_ascii=False, indent=2)
            pbar.update(1)

    return translated_rows


def unload_workers(workers: List[Dict[str, Any]]) -> None:
    for worker in workers:
        model = worker.get("model")
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        worker.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def filter_model_specs(only_models: Optional[List[str]]) -> List[ModelSpec]:
    selected = MODEL_SPECS
    if not INCLUDE_NIUTRANS_MODELS:
        selected = [spec for spec in selected if "NiuTrans/LMT-60-" not in spec.model_id]
    if not only_models:
        return selected
    lowered = [item.lower() for item in only_models]
    return [spec for spec in selected if any(token in spec.model_id.lower() for token in lowered)]


def save_result(output_dir: Path, model_id: str, translated_rows: List[Dict[str, Any]]) -> Path:
    output_path = output_dir / f"experiment_{safe_model_name(model_id)}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(translated_rows, f, ensure_ascii=False, indent=2)
    return output_path


def save_error(output_dir: Path, model_id: str, error: Exception) -> Path:
    output_path = output_dir / f"experiment_{safe_model_name(model_id)}.json"
    payload = {
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_FILE)
    selected_models = filter_model_specs(ONLY_MODELS)
    if not selected_models:
        raise ValueError("No models selected. Check ONLY_MODELS and INCLUDE_NIUTRANS_MODELS.")

    gpu_ids, max_memory = build_gpu_memory_plan()
    dtype = preferred_dtype()

    print(f"Dataset file: {DATASET_FILE}")
    print(f"Total samples: {len(dataset)}")
    print("Iterations per sample: 1")
    print(f"Include NiuTrans models: {INCLUDE_NIUTRANS_MODELS}")
    print(f"Selected models: {len(selected_models)}")
    print(f"GPU ids used: {gpu_ids if gpu_ids else ['cpu-only']}")
    print(f"Max memory plan: {max_memory}")

    offload_root = output_dir / "offload"
    offload_root.mkdir(parents=True, exist_ok=True)

    for index, spec in enumerate(selected_models, start=1):
        print("=" * 100)
        print(f"[{index}/{len(selected_models)}] Running: {spec.model_id} ({spec.params_b}B)")
        workers: List[Dict[str, Any]] = []
        output_path = output_dir / f"experiment_{safe_model_name(spec.model_id)}.json"
        try:
            workers = load_workers_for_model(
                spec=spec,
                gpu_ids=gpu_ids,
                dtype=dtype,
                max_memory=max_memory,
                offload_root=offload_root,
            )
            translated_rows = translate_dataset_for_model(spec, workers, dataset, output_path)
            result_path = save_result(output_dir, spec.model_id, translated_rows)
            print(f"Saved: {result_path}")
        except Exception as exc:
            error_path = save_error(output_dir, spec.model_id, exc)
            print(f"Failed model {spec.model_id}: {exc}")
            print(f"Saved error report: {error_path}")
        finally:
            unload_workers(workers)

    print("=" * 100)
    print("All experiments completed.")


if __name__ == "__main__":
    main()
