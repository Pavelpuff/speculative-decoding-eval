import argparse
import os
import time

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loader import load_wikitext_split
from src.models.draft import BigramDraftModel, TransformerDraftWrapper
from src.models.generation import autoregressive_generate, speculative_generate_tensor
from src.utils.device import get_device
from src.utils.seed import set_seed


def sync(device: torch.device) -> None:
    """Synchronizes device for accurate time measurement."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if "HF_TOKEN" not in os.environ:
        raise ValueError("Please set the HF_TOKEN environment variable to access Gemma models.")

    wandb.init(project="speculative-decoding", config=vars(args))
    print(f"[*] Initializing evaluation on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=dtype).to(
        device
    )
    target_model.eval()

    draft_model_hf = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=dtype).to(
        device
    )
    draft_model_hf.eval()

    print("[*] Loading datasets...")
    train_ds = load_wikitext_split("train", tokenizer)
    test_ds = load_wikitext_split("test", tokenizer)
    samples = test_ds[: args.num_samples]

    print("[*] Training Bigram Draft Model on train split...")
    train_ids = [tok for example in train_ds for tok in example]
    fallback_token = tokenizer.encode("the", add_special_tokens=False)[0]
    bigram_draft_fn = BigramDraftModel(train_ids, fallback_token, device)

    # 1. Baseline Autoregressive
    print(f"\\n[*] Running Baseline Autoregressive (Context: {args.context_len})...")
    sync(device)
    start_time = time.perf_counter()
    for sample in tqdm(samples, desc="Baseline", leave=False):
        _ = autoregressive_generate(sample[: args.context_len], target_model, args.n_tokens, device)
    sync(device)
    baseline_time = time.perf_counter() - start_time
    wandb.log({"baseline/total_time_s": baseline_time})

    # 2. Bigram Speculative Decoding
    print("[*] Running Bigram Speculative Decoding...")
    total_proposed_bg, total_accepted_bg = 0, 0
    sync(device)
    start_bg_time = time.perf_counter()
    for sample in tqdm(samples, desc="Bigram Spec", leave=False):
        _, stats = speculative_generate_tensor(
            sample[: args.context_len], bigram_draft_fn, target_model, args.n_tokens, args.k, device
        )
        total_proposed_bg += stats["total_draft_proposed"]
        total_accepted_bg += stats["total_accepted"]
    sync(device)
    bg_time = time.perf_counter() - start_bg_time

    bg_acc_rate = total_accepted_bg / total_proposed_bg if total_proposed_bg > 0 else 0
    wandb.log(
        {
            "bigram_draft/acceptance_rate": bg_acc_rate,
            "bigram_draft/total_time_s": bg_time,
            "bigram_draft/speedup": baseline_time / bg_time,
        }
    )

    # 3. Context Length Experiment for Transformer Draft
    transformer_draft_fn = TransformerDraftWrapper(draft_model_hf)
    context_lengths = [10, 20, 50, 100]

    print("\\n[*] Running Transformer Speculative Decoding across context lengths...")
    for cl in context_lengths:
        total_proposed, total_accepted = 0, 0
        sync(device)
        start_time = time.perf_counter()

        for sample in tqdm(samples, desc=f"Transformer (Ctx={cl})", leave=False):
            _, stats = speculative_generate_tensor(
                sample[:cl], transformer_draft_fn, target_model, args.n_tokens, args.k, device
            )
            total_proposed += stats["total_draft_proposed"]
            total_accepted += stats["total_accepted"]
        sync(device)
        transformer_time = time.perf_counter() - start_time

        acc_rate = total_accepted / total_proposed if total_proposed > 0 else 0
        speedup = baseline_time / transformer_time if transformer_time > 0 else 0

        print(f"   Context: {cl:3d} | Acc Rate: {acc_rate:.2%} | Time: {transformer_time:.2f}s")

        wandb.log(
            {
                "transformer_draft/context_length": cl,
                "transformer_draft/acceptance_rate": acc_rate,
                "transformer_draft/total_time_s": transformer_time,
                "transformer_draft/speedup": speedup,
            }
        )

    wandb.finish()
    print("\\n[*] Evaluation complete. Metrics uploaded to W&B.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Speculative Decoding")
    parser.add_argument("--target_model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--draft_model", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--n_tokens", type=int, default=40)
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
