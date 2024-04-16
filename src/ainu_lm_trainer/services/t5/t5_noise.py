import random
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase

# borrowed from: https://github.com/lassl/lassl/blob/28911c74336e5a07d2a97f8d661fe7c1e71fa1d3/src/lassl/utils.py


def random_spans_noise_mask(
    noise_density: float, mean_span_length: float, length: int
) -> torch.Tensor:
    orig_len = length
    length = max(length, 2)
    num_noise_tokens = round(noise_density * length)
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = round(num_noise_tokens / mean_span_length)
    num_noise_spans = max(num_noise_spans, 1)  # set minumum to 1
    num_nonnoise_tokens = length - num_noise_tokens

    def _random_segmentation(num_items: int, num_segments: int) -> torch.Tensor:
        bars = torch.arange(num_items - 1) < num_segments - 1
        bars = bars[torch.randperm(bars.size(0))]
        bars = torch.cat((torch.tensor([0]), bars), dim=0)  # to make segment 0 nonzero
        segment_id = torch.cumsum(bars, dim=0)
        segment_length = torch.zeros(num_segments, dtype=torch.long).scatter_add(
            0, segment_id, torch.ones_like(segment_id)
        )
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = torch.stack(
        (nonnoise_span_lengths, noise_span_lengths), dim=1
    ).reshape(-1)
    span_starts = torch.cumsum(interleaved_span_lengths, dim=0)[:-1]
    span_start_indicator = (
        torch.zeros(length).long().scatter(0, span_starts, torch.ones_like(span_starts))
    )
    span_num = torch.cumsum(span_start_indicator, dim=0)
    is_noise = span_num % 2 == 1
    return is_noise[:orig_len]


def noise_span_to_unique_sentinel(
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[int] | torch.Tensor,
    noise_mask: torch.BoolTensor,
    first_sentinel_index: int,
    append_last_sentinel: bool = False,
    denoiser_prefix: Optional[str] = None,
    is_sentinel_index_descending: bool = True,
) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)

    if len(tokens) > len(noise_mask):
        offset = len(tokens) - len(noise_mask)
        random.seed(tokens[0].item())
        start_idx = random.randint(0, offset)
        tokens = tokens[start_idx : start_idx + len(noise_mask)]
        assert len(tokens) == len(noise_mask)

    prev_token_is_noise = torch.cat((torch.tensor([0]), noise_mask[:-1]), dim=0).bool()
    first_noise_tokens = torch.logical_and(
        noise_mask, torch.logical_not(prev_token_is_noise)
    )
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

    if is_sentinel_index_descending:
        sentinel = (
            first_sentinel_index + 1 - torch.cumsum(first_noise_tokens.long(), dim=0)
        )
    else:
        sentinel = (
            first_sentinel_index - 1 + torch.cumsum(first_noise_tokens.long(), dim=0)
        )
    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    ret = torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))

    if append_last_sentinel and is_sentinel_index_descending:
        last_sentinel_id = sentinel.min().reshape(-1) - 1
        ret = torch.cat((ret, last_sentinel_id), dim=0)
    elif append_last_sentinel and not is_sentinel_index_descending:
        last_sentinel_id = sentinel.max().reshape(-1) + 1
        ret = torch.cat((ret, last_sentinel_id), dim=0)
    ret = torch.cat(
        (ret, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)), dim=0
    )

    if denoiser_prefix:
        denoiser_prefix_enc = torch.tensor(
            tokenizer.encode(denoiser_prefix)[:1], dtype=torch.long
        )
        ret = torch.cat((denoiser_prefix_enc, ret), dim=0)
    return ret
