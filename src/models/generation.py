import torch
import torch.nn as nn


@torch.no_grad()
def autoregressive_generate(
    context_ids: list[int], model: nn.Module, n_tokens: int, device: torch.device
) -> list[int]:
    """Standard autoregressive generation (baseline)."""
    input_ids = torch.tensor([context_ids], device=device)
    for _ in range(n_tokens):
        logits = model(input_ids).logits
        next_tok = logits[:, -1:].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    return input_ids[0, len(context_ids) :].tolist()


@torch.no_grad()
def verify_draft_tensor(
    context_ids: torch.Tensor, draft_ids: torch.Tensor, model: nn.Module
) -> torch.Tensor:
    """Fully vectorized verification of draft tokens."""
    n = context_ids.shape[0]
    k = draft_ids.shape[0]

    input_ids = torch.cat((context_ids, draft_ids), dim=0).unsqueeze(0)
    logits = model(input_ids).logits.squeeze(0)

    draft_token_logits = logits[n - 1 : n + k - 1]
    next_token_logits = logits[n + k - 1 : n + k]

    preds = torch.cat((draft_token_logits, next_token_logits), dim=0).argmax(dim=-1)
    target_preds_for_draft = preds[:k]

    mismatches = (draft_ids != target_preds_for_draft).nonzero(as_tuple=True)[0]
    last_accepted_idx = mismatches[0].item() if mismatches.numel() != 0 else k

    accepted_draft_ids = draft_ids[:last_accepted_idx]
    next_token_id = preds[last_accepted_idx : last_accepted_idx + 1]

    return torch.cat((context_ids, accepted_draft_ids, next_token_id), dim=0)


@torch.no_grad()
def speculative_generate_tensor(
    context_ids: list[int], draft_fn, model, n_tokens: int, k: int, device: torch.device
) -> tuple[list[int], dict]:
    curr_ids = torch.tensor(context_ids, device=device)
    context_len = curr_ids.shape[0]

    stats = {"total_draft_proposed": 0, "total_accepted": 0, "verification_rounds": 0}

    while curr_ids.shape[0] - context_len < n_tokens:
        remaining = n_tokens - (curr_ids.shape[0] - context_len)

        current_k = min(k, remaining)

        draft_ids = draft_fn(curr_ids, current_k)

        old_len = curr_ids.shape[0]
        curr_ids = verify_draft_tensor(curr_ids, draft_ids, model)
        new_len = curr_ids.shape[0]

        added = new_len - old_len
        accepted_draft_tokens = added - 1
        kept = min(added, remaining)

        stats["verification_rounds"] += 1
        stats["total_draft_proposed"] += current_k
        stats["total_accepted"] += min(accepted_draft_tokens, kept)

        if new_len > context_len + n_tokens:
            curr_ids = curr_ids[: context_len + n_tokens]

    generated_token_ids = curr_ids[context_len : context_len + n_tokens].tolist()
    return generated_token_ids, stats
