from collections import defaultdict

import torch
import torch.nn as nn


class BigramDraftModel:
    """Naive bigram-based draft model trained on the dataset."""

    def __init__(self, train_ids: list[int], fallback_token: int, device: torch.device):
        self.device = device
        self.fallback_token = fallback_token
        self.bigram_next: dict[int, int] = self._train_bigrams(train_ids)

    def _train_bigrams(self, train_ids: list[int]) -> dict[int, int]:
        bigram_counts = defaultdict(lambda: defaultdict(int))
        for id1, id2 in zip(train_ids, train_ids[1:]):
            bigram_counts[id1][id2] += 1

        bigram_next = {}
        for tok, successors in bigram_counts.items():
            bigram_next[tok] = max(successors, key=successors.get)
        return bigram_next

    def __call__(self, context_ids: torch.Tensor, k: int) -> torch.Tensor:
        draft = []
        last_tok = context_ids[-1].item()
        for _ in range(k):
            next_tok = self.bigram_next.get(last_tok, self.fallback_token)
            draft.append(next_tok)
            last_tok = next_tok
        return torch.tensor(draft, device=self.device)


class TransformerDraftWrapper:
    """Wrapper for HuggingFace CausalLM models to act as draft models."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def __call__(self, context_ids: torch.Tensor, k: int) -> torch.Tensor:
        input_ids = context_ids.unsqueeze(0)
        for _ in range(k):
            logits = self.model(input_ids).logits
            next_token = logits[:, -1:].argmax(dim=-1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids[0, -k:]
