import gc
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, SeamlessM4Tv2Model

INPUT_FILE = Path("Outputs/All Models Test/merged_translations.json")
OUTPUT_ROOT = Path("/ssd_scratch/phani_ssd/model_out")
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints" / "similarities"
CHECKPOINT_EVERY = 100

MODEL_QWEN = "Qwen/Qwen3-Embedding-8B"
MODEL_GOOGLE = "google/embeddinggemma-300m"
MODEL_FACEBOOK = "facebook/seamless-m4t-v2-large"

QWEN_MAX_LENGTH = 8192
FACEBOOK_MAX_LENGTH = 1024


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
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


def mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    summed = (last_hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


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


def safe_model_tag(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "model"


def per_model_output_path(model_name: str) -> Path:
    return OUTPUT_ROOT / f"similarity_{safe_model_tag(model_name)}.json"


def checkpoint_path(model_name: str, processed_count: int) -> Path:
    return CHECKPOINT_DIR / (
        f"similarity_{safe_model_tag(model_name)}_checkpoint_{processed_count:04d}.json"
    )


def merged_output_path() -> Path:
    return OUTPUT_ROOT / "merged_similarities.json"


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_merged_samples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, found {type(data).__name__}.")

    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            print(f"[warn] Skipping row {idx} because it is not a JSON object.")
            continue

        utterence = row.get("utterence")
        if not isinstance(utterence, dict):
            print(f"[warn] Row {idx} has non-dict 'utterence'. Replacing with empty dict.")
            utterence = {}

        samples.append(
            {
                "input": "" if row.get("input") is None else str(row.get("input")),
                "utterence": dict(utterence),
                "label": row.get("label"),
            }
        )
    return samples


def build_base_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample in samples:
        records.append(
            {
                "input": sample["input"],
                "utterence": sample["utterence"],
                "similarity": {},
                "label": sample.get("label"),
            }
        )
    return records


def build_failed_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = build_base_records(samples)
    for idx, sample in enumerate(samples):
        utterence_map = sample.get("utterence", {})
        if isinstance(utterence_map, dict):
            records[idx]["similarity"] = {str(k): None for k in utterence_map.keys()}
    return records


def cosine_similarity_from_normalized(a: Tensor, b: Tensor) -> float:
    return float(torch.dot(a, b).item())


def process_samples_for_model(
    model_name: str,
    samples: List[Dict[str, Any]],
    encode_input_fn: Callable[[str], Tensor],
    encode_utterence_fn: Callable[[str], Tensor],
) -> List[Dict[str, Any]]:
    records = build_base_records(samples)

    for idx, sample in enumerate(
        tqdm(samples, desc=f"Similarity with {model_name}", unit="sample"), start=1
    ):
        utterence_map = sample.get("utterence", {})
        similarity_map: Dict[str, Optional[float]] = {}
        input_embedding: Optional[Tensor] = None

        try:
            input_embedding = encode_input_fn(sample.get("input", ""))
        except Exception as exc:
            print(f"[warn] {model_name}: input embedding failed at sample {idx - 1}: {exc}")

        if isinstance(utterence_map, dict):
            for translated_model_name, translated_text in utterence_map.items():
                key = str(translated_model_name)

                if input_embedding is None:
                    similarity_map[key] = None
                    continue

                try:
                    utterence_embedding = encode_utterence_fn(
                        "" if translated_text is None else str(translated_text)
                    )
                    similarity_map[key] = cosine_similarity_from_normalized(
                        input_embedding, utterence_embedding
                    )
                except Exception as exc:
                    print(
                        f"[warn] {model_name}: utterence embedding failed at sample "
                        f"{idx - 1}, model '{key}': {exc}"
                    )
                    similarity_map[key] = None
        else:
            print(
                f"[warn] {model_name}: sample {idx - 1} has invalid 'utterence'; "
                "using empty similarity map."
            )

        records[idx - 1]["similarity"] = similarity_map

        if idx % CHECKPOINT_EVERY == 0:
            cp_path = checkpoint_path(model_name, idx)
            save_json(records[:idx], cp_path)
            print(f"Saved checkpoint: {cp_path}")

    return records


def load_qwen_components() -> tuple[AutoTokenizer, AutoModel, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN, padding_side="left")
    load_kwargs: Dict[str, Any] = {"torch_dtype": preferred_dtype()}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(MODEL_QWEN, **load_kwargs)
    model.eval()
    input_device = get_model_input_device(model)
    return tokenizer, model, input_device


def encode_qwen_text(
    text: str, tokenizer: AutoTokenizer, model: AutoModel, input_device: torch.device
) -> Tensor:
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


def run_qwen_model(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModel] = None
    input_device = torch.device("cpu")

    try:
        tokenizer, model, input_device = load_qwen_components()

        def encode_text(text: str) -> Tensor:
            assert tokenizer is not None and model is not None
            return encode_qwen_text(text, tokenizer, model, input_device)

        records = process_samples_for_model(
            model_name="Qwen",
            samples=samples,
            encode_input_fn=encode_text,
            encode_utterence_fn=encode_text,
        )
    except Exception as exc:
        print(f"[warn] Qwen: failed to run model. Reason: {exc}")
        records = build_failed_records(samples)
    finally:
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        cleanup_memory()

    output_path = per_model_output_path("Qwen")
    save_json(records, output_path)
    print(f"Saved Qwen similarities to: {output_path}")
    return records


def load_google_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for google/embeddinggemma-300m. "
            "Install with `pip install sentence-transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_GOOGLE, device=device)


def encode_google_text(text: str, model: Any) -> Tensor:
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embedding_tensor = torch.as_tensor(embedding, dtype=torch.float32).flatten()
    norm = embedding_tensor.norm(p=2)
    if norm.item() > 0:
        embedding_tensor = embedding_tensor / norm
    return embedding_tensor.cpu()


def run_google_model(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model: Any = None
    try:
        model = load_google_model()

        def encode_text(text: str) -> Tensor:
            assert model is not None
            return encode_google_text(text, model)

        records = process_samples_for_model(
            model_name="Google",
            samples=samples,
            encode_input_fn=encode_text,
            encode_utterence_fn=encode_text,
        )
    except Exception as exc:
        print(f"[warn] Google: failed to run model. Reason: {exc}")
        records = build_failed_records(samples)
    finally:
        if model is not None:
            del model
        cleanup_memory()

    output_path = per_model_output_path("Google")
    save_json(records, output_path)
    print(f"Saved Google similarities to: {output_path}")
    return records


def seamless_encoder_hidden_states(
    model: SeamlessM4Tv2Model, input_ids: Tensor, attention_mask: Tensor
) -> Tensor:
    if hasattr(model, "get_encoder"):
        encoder = model.get_encoder()
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state

    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", 0)
    decoder_input_ids = torch.full(
        (input_ids.size(0), 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        return_dict=True,
    )
    if getattr(outputs, "encoder_last_hidden_state", None) is not None:
        return outputs.encoder_last_hidden_state
    return outputs.last_hidden_state


def load_facebook_components() -> tuple[AutoProcessor, SeamlessM4Tv2Model, torch.device]:
    processor = AutoProcessor.from_pretrained(MODEL_FACEBOOK)
    load_kwargs: Dict[str, Any] = {"torch_dtype": preferred_dtype()}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    model = SeamlessM4Tv2Model.from_pretrained(MODEL_FACEBOOK, **load_kwargs)
    model.eval()
    input_device = get_model_input_device(model)
    return processor, model, input_device


def encode_facebook_text(
    text: str,
    processor: AutoProcessor,
    model: SeamlessM4Tv2Model,
    input_device: torch.device,
    src_lang: Optional[str],
) -> Tensor:
    processor_kwargs: Dict[str, Any] = {
        "text": text,
        "padding": True,
        "truncation": True,
        "max_length": FACEBOOK_MAX_LENGTH,
        "return_tensors": "pt",
    }
    if src_lang is not None:
        processor_kwargs["src_lang"] = src_lang

    try:
        batch = processor(**processor_kwargs)
    except Exception:
        processor_kwargs.pop("src_lang", None)
        batch = processor(**processor_kwargs)

    batch = {k: v.to(input_device) for k, v in batch.items()}
    with torch.no_grad():
        hidden_states = seamless_encoder_hidden_states(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        embedding = mean_pool(hidden_states, batch["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding[0].float().cpu()


def run_facebook_model(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor: Optional[AutoProcessor] = None
    model: Optional[SeamlessM4Tv2Model] = None
    input_device = torch.device("cpu")

    try:
        processor, model, input_device = load_facebook_components()

        def encode_input(text: str) -> Tensor:
            assert processor is not None and model is not None
            return encode_facebook_text(text, processor, model, input_device, src_lang="eng")

        def encode_utterence(text: str) -> Tensor:
            assert processor is not None and model is not None
            return encode_facebook_text(text, processor, model, input_device, src_lang="tel")

        records = process_samples_for_model(
            model_name="Facebook",
            samples=samples,
            encode_input_fn=encode_input,
            encode_utterence_fn=encode_utterence,
        )
    except Exception as exc:
        print(f"[warn] Facebook: failed to run model. Reason: {exc}")
        records = build_failed_records(samples)
    finally:
        if processor is not None:
            del processor
        if model is not None:
            del model
        cleanup_memory()

    output_path = per_model_output_path("Facebook")
    save_json(records, output_path)
    print(f"Saved Facebook similarities to: {output_path}")
    return records


def merge_similarity_records(
    samples: List[Dict[str, Any]], model_outputs: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        row: Dict[str, Any] = {
            "input": sample["input"],
            "utterence": sample["utterence"],
            "similarity": {},
            "label": sample.get("label"),
        }

        for model_name, records in model_outputs.items():
            if idx >= len(records):
                print(
                    f"[warn] {model_name}: missing row {idx} during merge. "
                    "Using empty similarity map."
                )
                row["similarity"][model_name] = {}
                continue

            similarity_map = records[idx].get("similarity", {})
            if not isinstance(similarity_map, dict):
                print(
                    f"[warn] {model_name}: row {idx} similarity is not a dict. "
                    "Using empty similarity map."
                )
                similarity_map = {}

            row["similarity"][model_name] = similarity_map

        merged.append(row)
    return merged


def main() -> None:
    samples = load_merged_samples(INPUT_FILE)
    print(f"Loaded {len(samples)} rows from {INPUT_FILE}")

    print(f"\nProcessing one-by-one with {MODEL_QWEN}...")
    qwen_records = run_qwen_model(samples)

    print(f"\nProcessing one-by-one with {MODEL_GOOGLE}...")
    google_records = run_google_model(samples)

    print(f"\nProcessing one-by-one with {MODEL_FACEBOOK}...")
    facebook_records = run_facebook_model(samples)

    merged = merge_similarity_records(
        samples=samples,
        model_outputs={
            "Qwen": qwen_records,
            "Google": google_records,
            "Facebook": facebook_records,
        },
    )
    merged_path = merged_output_path()
    save_json(merged, merged_path)
    print(f"\nSaved merged similarities to: {merged_path}")


if __name__ == "__main__":
    main()
