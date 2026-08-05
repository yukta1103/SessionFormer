import torch

from sessionformer.data.vocab import PAD_IDX, UNK_IDX


@torch.no_grad()
def mine_hard_negatives(model, loader, device, top_k: int = 50, num_hard_negatives: int = 5) -> list[list[int]]:
    """For each session's last valid position, takes SASRec's top_k
    highest-scoring items and keeps the ones that are NOT the true target as
    hard negatives. loader must not shuffle: the returned list is aligned
    with the underlying dataset's iteration order."""
    model.eval()
    all_hard_negatives: list[list[int]] = []
    for input_seq, target_seq, lengths in loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        hidden = model.encode_sequence(input_seq)

        batch_idx = torch.arange(input_seq.size(0), device=device)
        last_idx = (lengths - 1).to(device)
        last_hidden = hidden[batch_idx, last_idx]
        target_last = target_seq[batch_idx, last_idx]

        scores = model.score_all_items(last_hidden)
        scores[:, PAD_IDX] = float("-inf")
        scores[:, UNK_IDX] = float("-inf")

        topk = scores.topk(top_k, dim=-1).indices
        for row, target in zip(topk.tolist(), target_last.tolist()):
            hard_negs = [item for item in row if item != target][:num_hard_negatives]
            all_hard_negatives.append(hard_negs)

    model.train()
    return all_hard_negatives
