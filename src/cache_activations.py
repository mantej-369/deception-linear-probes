"""Step 2: Frozen-weight residual stream activation caching with trailing-token masking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compute_assistant_span(
    prompt_len: int,
    total_len: int,
    exclude_last_n: int = 0,
) -> tuple[int, int]:
    """Return [start, end) token indices for the assistant span after trailing masking."""
    if prompt_len < 0 or total_len <= prompt_len:
        raise ValueError(f"Invalid sequence lengths: prompt_len={prompt_len}, total_len={total_len}")
    if exclude_last_n < 0:
        raise ValueError(f"exclude_last_n must be >= 0, got {exclude_last_n}")

    end = total_len - exclude_last_n
    if end <= prompt_len:
        raise ValueError(
            f"Token starvation: assistant span has only {total_len - prompt_len} tokens, "
            f"cannot mask last {exclude_last_n} tokens."
        )
    return prompt_len, end


def flatten_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Normalise both contrastive pair files and single-turn control files into uniform rows."""
    raw_rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    items: list[dict[str, Any]] = []

    for row in raw_rows:
        if "honest" in row and "deceptive" in row:
            mask_n = int(row.get("exclude_last_n_tokens", 5))
            for side, label in (("honest", 0), ("deceptive", 1)):
                block = row[side]
                items.append(
                    {
                        "id": f"{row['pair_id']}_{side}",
                        "pair_id": row["pair_id"],
                        "category": row["category"],
                        "system": block["system"],
                        "user": block["user"],
                        "assistant": block["assistant_prefix"],
                        "exclude_last_n": mask_n,
                        "label": label,
                    }
                )
        else:
            items.append(
                {
                    "id": row["id"],
                    "pair_id": row["id"],
                    "category": row.get("subtype", row.get("tier", "control")),
                    "system": row.get("system", ""),
                    "user": row["user"],
                    "assistant": row["assistant"],
                    "exclude_last_n": int(row.get("exclude_last_n_tokens", 0)),
                    "label": int(row["label"]),
                }
            )
    return items


def extract_dataset_activations(
    model_name: str,
    input_path: Path,
    output_path: Path,
    dtype_str: str = "float16",
) -> None:
    """Run frozen forward passes and save per-layer mean and last-token residual activations."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype_str]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device} ({dtype_str})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    items = flatten_jsonl_records(input_path)
    mean_acts, last_acts, span_lengths = [], [], []
    labels, ids, categories = [], [], []

    with torch.inference_mode():
        for idx, item in enumerate(items, start=1):
            messages = []
            if item["system"]:
                messages.append({"role": "system", "content": item["system"]})
            messages.append({"role": "user", "content": item["user"]})

            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + item["assistant"]

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            start, end = compute_assistant_span(
                prompt_len=len(prompt_ids),
                total_len=len(full_ids),
                exclude_last_n=item["exclude_last_n"],
            )

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=model.device)
            outputs = model(input_ids=input_tensor, output_hidden_states=True, use_cache=False)

            # hidden_states[0] is token embeddings; [1:] are transformer block residual outputs
            layer_means = []
            layer_lasts = []
            for hidden in outputs.hidden_states[1:]:
                span = hidden[0, start:end, :]  # (span_len, d_model)
                layer_means.append(span.mean(dim=0).to(torch.float16).cpu())
                layer_lasts.append(span[-1, :].to(torch.float16).cpu())

            mean_acts.append(torch.stack(layer_means, dim=0))  # (num_layers, d_model)
            last_acts.append(torch.stack(layer_lasts, dim=0))
            span_lengths.append(end - start)
            labels.append(item["label"])
            ids.append(item["id"])
            categories.append(item["category"])

            if idx % 20 == 0 or idx == len(items):
                print(f"Processed {idx}/{len(items)} items (last span={end - start} tokens)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "source_file": str(input_path),
        "mean_activations": torch.stack(mean_acts, dim=0),  # (N, L, D)
        "last_token_activations": torch.stack(last_acts, dim=0),  # (N, L, D)
        "labels": torch.tensor(labels, dtype=torch.long),
        "span_lengths": span_lengths,
        "ids": ids,
        "categories": categories,
    }
    torch.save(payload, output_path)
    print(f"Saved activation cache to {output_path} with shape {tuple(payload['mean_activations'].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache residual stream activations for probing.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    extract_dataset_activations(
        model_name=args.model,
        input_path=args.input,
        output_path=args.output,
        dtype_str=args.dtype,
    )


if __name__ == "__main__":
    main()