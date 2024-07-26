import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

from ....config import (
    DatasetConfig,
    TrainingConfig,
    WorkspaceConfig,
)


def train(
    config_dataset: DatasetConfig,
    config_training: TrainingConfig,
    config_workspace: WorkspaceConfig,
    tokenizer_name: str,
    context_length: int = 128,
) -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    config = GPT2Config.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)
    model = model.to("cuda") if torch.cuda.is_available() else model

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset_dict = config_dataset.load()
    dataset_dict = dataset_dict.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=context_length,
            padding="max_length",
            return_tensors="pt",
        ),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir=str(config_workspace.checkpoint_dir),
        logging_dir=str(config_workspace.logging_dir),
        report_to=["tensorboard"],
    )
    training_args = config_training.extend(training_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
    )

    trainer.train()
    trainer.save_model(str(config_workspace.model_dir))

    if config_training.push_to_hub:
        model.push_to_hub(training_args.hub_model_id)
        tokenizer.push_to_hub(training_args.hub_model_id)
