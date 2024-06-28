import evaluate
import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
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

metric = evaluate.load("sacrebleu")


class Mt5Trainer:
    __context_length = 128
    __model_name = "aynumosir/mt5-small-ainu"
    __task_prefix = "translate: Japanese to Ainu: "

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
        dataset = dataset.filter(
            lambda example: len(example["sentence"]) > 0
            and len(example["translation"]) > 0
        )
        # 文語に括弧書きで口語がある資料。除外する。
        dataset = dataset.filter(
            lambda example: not (
                example["book"] == "鍋沢元蔵筆録ノート"
                and example["title"] == "kamuyyukar-2"
            )
        )

        # https://huggingface.co/docs/transformers/en/tasks/summarization#preprocess
        def preprocess(examples: dict) -> dict:
            inputs = tokenizer(
                [self.__task_prefix + text for text in examples["translation"]],
                text_target=examples["sentence"],
                max_length=self.__context_length,
                truncation=True,
            )
            return inputs

        dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
        )

        dataset_dict = dataset.train_test_split(test_size=0.1)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            save_strategy="epoch",
            evaluation_strategy="epoch",
            output_dir=str(self.__workspace_config.checkpoint_dir),
            generation_max_length=self.__context_length,
            predict_with_generate=True,
            load_best_model_at_end=True,
            # https://ethen8181.github.io/machine-learning/deep_learning/seq2seq/translation_mt5/translation_mt5.html
            learning_rate=6e-4,
            weight_decay=0.01,
            metric_for_best_model="bleu",
            greater_is_better=True,
            # gradient_accumulation_steps=2,
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
            compute_metrics=lambda pred: self.__compute_metrics(tokenizer, pred),
        )

        trainer.train()
        trainer.save_model(str(self.__workspace_config.model_dir))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)

        if self.__training_config.push_to_hub:
            model.push_to_hub(self.__model_name)
            tokenizer.push_to_hub(self.__model_name)

    def __compute_metrics(
        self, tokenizer: MT5Tokenizer, eval_preds: EvalPrediction
    ) -> dict:
        predictions, labels = eval_preds

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}
