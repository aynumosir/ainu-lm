import evaluate
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ....config import (
    DatasetConfig,
    FineTuningConfig,
    TrainingConfig,
    WorkspaceConfig,
)

chrf = evaluate.load("chrf")


def compute_metrics(tokenizer: AutoTokenizer, eval_preds: EvalPrediction) -> dict:
    predictions, labels = eval_preds

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    metric = chrf.compute(predictions=decoded_preds, references=decoded_labels)
    return {"chrf": metric["score"]}


def train(
    config_dataset: DatasetConfig,
    config_training: TrainingConfig,
    config_workspace: WorkspaceConfig,
    config_fine_tuning: FineTuningConfig,
    context_length: int = 128,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config_fine_tuning.base_tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(config_fine_tuning.base_model)
    model = model.to("cuda") if torch.cuda.is_available() else model

    dataset_dict = config_dataset.load()

    dataset_dict = dataset_dict.map(
        lambda examples: tokenizer(
            examples["kana"],
            text_target=examples["text"],
            max_length=context_length,
            truncation=True,
        ),
        batched=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        save_strategy="epoch",
        evaluation_strategy="epoch",
        output_dir=str(config_workspace.checkpoint_dir),
        generation_max_length=context_length,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        logging_dir=str(config_workspace.logging_dir),
        report_to=["tensorboard"],
    )
    training_args = config_training.extend(training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=lambda eval_preds: compute_metrics(tokenizer, eval_preds),
    )

    trainer.train()
    trainer.save_model(str(config_workspace.model_dir))

    if config_training.push_to_hub:
        model.push_to_hub(training_args.hub_model_id)
        tokenizer.push_to_hub(training_args.hub_model_id)
