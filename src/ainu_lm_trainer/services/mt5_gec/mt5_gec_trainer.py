import torch
from datasets import interleave_datasets, load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ...config import (
    FineTuningConfig,
    TrainingConfig,
    WorkspaceConfig,
)


class Mt5GecTrainer:
    __context_length = 128
    __model_name = "aynumosir/mt5-small-ainu-gec"

    __fine_tuning_config: FineTuningConfig
    __training_config: TrainingConfig
    __workspace_config: WorkspaceConfig

    def __init__(
        self,
        fine_tuning_config: FineTuningConfig,
        training_config: TrainingConfig,
        workspace_config: WorkspaceConfig,
    ) -> None:
        self.__fine_tuning_config = fine_tuning_config
        self.__training_config = training_config
        self.__workspace_config = workspace_config

    def train(self) -> None:
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        model = model.to("cuda") if torch.cuda.is_available() else model

        dataset = interleave_datasets(
            [
                load_dataset("aynumosir/ainu-gec-rule-based", split="train"),
                load_dataset("aynumosir/ainu-gec-back-translation", split="train"),
            ],
            probabilities=[0.5, 0.5],
            stopping_strategy="first_exhausted",
        )

        # https://huggingface.co/docs/transformers/en/tasks/summarization#preprocess
        dataset = dataset.map(
            lambda examples: tokenizer(
                [
                    self.__get_task_prefix(example) + example["text"]
                    for example in examples
                ],
                text_target=[text for text in examples["target"]],
                max_length=self.__context_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            batched=True,
        )

        dataset_dict = dataset.train_test_split(test_size=0.1)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            save_strategy="epoch",
            evaluation_strategy="epoch",
            output_dir=str(self.__workspace_config.checkpoint_dir),
            generation_max_length=self.__context_length,
            predict_with_generate=True,
            # とくに根拠はないけどT5の事前学習でうまく行った数値
            learning_rate=3e-4,
            weight_decay=0.01,
            # https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/#toc-5
            load_best_model_at_end=True,
            logging_dir=str(self.__workspace_config.logging_dir),
            report_to=["tensorboard"],
        )
        training_args = self.__training_config.extend(training_args)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(str(self.__workspace_config.model_dir))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)

        if self.__training_config.push_to_hub:
            model.push_to_hub(self.__model_name)
            tokenizer.push_to_hub(self.__model_name)

    def __get_task_prefix(self, example: dict) -> str:
        if example["dialect"] is not None:
            return f"fix Ainu sentence ({example['dialect']}, {example['pronoun']}): "
        else:
            return f"fix Ainu sentence (沙流, {example['pronoun']}): "
