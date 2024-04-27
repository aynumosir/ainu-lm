import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ...config import (
    DatasetConfig,
    FineTuningConfig,
    TrainingConfig,
    WorkspaceConfig,
)


class Mt5GecTrainer:
    __context_length = 128
    __model_name = "aynumosir/mt5-small-ainu-gec"
    __task_prefix = "fix Ainu sentence: "

    __dataset_config: DatasetConfig
    __fine_tuning_config: FineTuningConfig
    __training_config: TrainingConfig
    __workspace_config: WorkspaceConfig

    def __init__(
        self,
        dataset_config: DatasetConfig,
        fine_tuning_config: FineTuningConfig,
        training_config: TrainingConfig,
        workspace_config: WorkspaceConfig,
    ) -> None:
        self.__dataset_config = dataset_config
        self.__fine_tuning_config = fine_tuning_config
        self.__training_config = training_config
        self.__workspace_config = workspace_config

    def train(self) -> None:
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        model = model.to("cuda") if torch.cuda.is_available() else model

        dataset = self.__dataset_config.load()
        dataset = dataset.map(
            lambda examples: self.__preprocess_function(tokenizer, examples),
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
            fp16=True if torch.cuda.is_available() else False,
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

    # https://huggingface.co/docs/transformers/en/tasks/summarization#preprocess
    def __preprocess_function(self, tokenizer: MT5Tokenizer, examples: dict) -> dict:
        inputs = tokenizer(
            [self.__task_prefix + text for text in examples["text"]],
            text_target=examples["labels"],
            max_length=self.__context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs
