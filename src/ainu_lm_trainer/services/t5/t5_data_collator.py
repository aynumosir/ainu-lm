import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import _torch_collate_batch
from transformers.models.bart.modeling_bart import shift_tokens_right

from .t5_noise import noise_span_to_unique_sentinel, random_spans_noise_mask


class DataCollatorForT5:
    """
    Processing training examples to mini-batch for T5
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int = 8,
        noise_density: float = 0.15,
        mean_span_length: float = 3.0,
        first_extra_id: str = "<extra_id_0>",
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self._decoder_start_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.pad_token_id
        )
        self.first_sentinel_index: int = self.tokenizer.get_vocab()[first_extra_id]
        self.descending_sentinel: bool = self.is_sentinel_index_descending(
            first_extra_id
        )

    # check whether sentinel tokens indices are in descending order or not
    def is_sentinel_index_descending(self, first_extra_id: str) -> bool:
        """
        The logic for "noise_span_to_unique_sentinel()" totally depends on the order of sentinel token ids indices in vocabs.
        So, it is necessary to check whether the vocab indices of <extra_id_0> to <extra_id_99> are increasing or decreasing.
        For example, if "<extra_id_0>" is mapped to 10 in vocab and "<extra_id_99>" is mapped to 109 in vocab,
        the output of is_sentinel_index_descending() should be "False" and vice versa.
        """
        if "0" in first_extra_id:
            last_extra_id = "99".join(first_extra_id.split("0"))
        else:
            raise ValueError("first sentinel tokens must include zero")
        return (
            self.tokenizer.get_vocab()[last_extra_id]
            - self.tokenizer.get_vocab()[first_extra_id]
            < 0
        )

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        example_n = len(examples)
        example_len = len(examples[0])
        noise_masks = [
            random_spans_noise_mask(
                self.noise_density, self.mean_span_length, example_len
            )
            for _ in range(example_n)
        ]
        inputs = [
            noise_span_to_unique_sentinel(
                self.tokenizer,
                example,
                noise_mask,
                first_sentinel_index=self.first_sentinel_index,
                is_sentinel_index_descending=self.descending_sentinel,
            )
            for example, noise_mask in zip(examples, noise_masks)
        ]
        targets = [
            noise_span_to_unique_sentinel(
                self.tokenizer,
                example,
                ~noise_mask,
                first_sentinel_index=self.first_sentinel_index,
                append_last_sentinel=True,
                is_sentinel_index_descending=self.descending_sentinel,
            )
            for example, noise_mask in zip(examples, noise_masks)
        ]
        # make labels and input_ids
        batch = {
            "input_ids": _torch_collate_batch(
                inputs,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=None,  # all samples' length are set to self.max_length by design
            ),
            "labels": _torch_collate_batch(
                targets,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=None,  # labels' length are all sample by design
            ),
        }
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self._decoder_start_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            batch["decoder_input_ids"] == self.tokenizer.pad_token_id,
            0,
            torch.ones_like(batch["decoder_input_ids"]),
        )
        batch["attention_mask"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id,
            0,
            torch.ones_like(batch["input_ids"]),
        )
        return batch
