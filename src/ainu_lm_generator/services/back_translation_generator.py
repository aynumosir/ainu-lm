import asyncio
import os

import aiohttp
from datasets import Dataset, load_dataset
from tqdm.asyncio import tqdm_asyncio
from transformers import MT5Tokenizer

TASK_PREFIX = "translate: Japanese to Ainu: "
CONTEXT_LENGTH = 128
NUM_RETURN_SEQUENCES = 20


async def query(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=1000,
    ) as response:
        return await response.json()


async def process_batch(
    session: aiohttp.ClientSession,
    url: str,
    examples: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async with semaphore:
        try:
            translations_list = await query(
                session,
                url,
                {
                    "inputs": [
                        TASK_PREFIX + translation
                        for translation in examples["translation"]
                    ],
                    "parameters": {
                        "max_length": CONTEXT_LENGTH,
                        "do_sample": True,
                        "num_return_sequences": NUM_RETURN_SEQUENCES,
                        "early_stopping": True,
                    },
                },
            )
            if "error" in translations_list:
                raise Exception(translations_list["error"])
        except asyncio.TimeoutError:
            print("asyncio.TimeoutError:", examples["translation"])
            return []
        except BaseException as e:
            print("Unexpected error occurred:", e, examples["translation"])
            return []

        entries: list[dict] = []

        for translations, target in zip(translations_list, examples["sentence"]):
            # Remove duplicates
            translations = list(
                set([translation["translation_text"] for translation in translations])
            )
            # Pair the translations with the target
            for translation in translations:
                entries.append({"text": translation, "target": target})

        return entries


async def _generate_back_translation(
    inference_endpoint_url: str, semaphore_count: int, batch_size: int
) -> list[dict]:
    entries = []

    # Setup the tokenizer
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    # Prepare the dataset
    dataset = load_dataset("aynumosir/ainu-corpora", split="train")
    dataset = dataset.filter(
        lambda example: len(example["sentence"]) > 0 and len(example["translation"]) > 0
    )
    dataset = dataset.filter(
        lambda example: len(tokenizer(example["translation"])) <= CONTEXT_LENGTH * 0.9
    )

    # Call the API
    semaphore = asyncio.Semaphore(semaphore_count)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_batch(session, inference_endpoint_url, examples, semaphore)
            for examples in dataset.iter(batch_size)
        ]
        for batched_entries in await tqdm_asyncio.gather(*tasks):
            entries.extend(batched_entries)

    return entries


def generate_back_translation(
    dataset_name: str,
    inference_endpoint_url: str,
    semaphore_count: int,
    batch_size: int,
    push_to_hub: bool = True,
) -> None:
    entries = asyncio.run(
        _generate_back_translation(inference_endpoint_url, semaphore_count, batch_size)
    )
    dataset = Dataset.from_list(entries)

    if push_to_hub:
        dataset.push_to_hub(dataset_name)
