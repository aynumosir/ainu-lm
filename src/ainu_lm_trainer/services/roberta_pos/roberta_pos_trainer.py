import evaluate
import numpy as np
import torch
from transformers import (
    EvalPrediction,
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ...config import (
    DatasetConfig,
    FineTuningConfig,
    TrainingConfig,
    WorkspaceConfig,
)
from .label_names import label_names

seqeval = evaluate.load("seqeval")


class RobertaPosTrainer:
    __context_length = 128
    __model_name = "aynumosir/roberta-base-ainu-pos"

    __fine_tuning_config: FineTuningConfig
    __dataset_config: DatasetConfig
    __training_config: TrainingConfig
    __workspace_config: WorkspaceConfig

    def __init__(
        self,
        fine_tuning_config: FineTuningConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        workspace_config: WorkspaceConfig,
    ) -> None:
        self.__fine_tuning_config = fine_tuning_config
        self.__dataset_config = dataset_config
        self.__training_config = training_config
        self.__workspace_config = workspace_config

    def train(self) -> None:
        tokenizer = RobertaTokenizerFast.from_pretrained(
            str(self.__fine_tuning_config.model),
            add_prefix_space=True,
        )

        model = RobertaForTokenClassification.from_pretrained(
            str(self.__fine_tuning_config.model),
            id2label={i: label for i, label in enumerate(label_names)},
            label2id={label: i for i, label in enumerate(label_names)},
        )
        model = model.to("cuda") if torch.cuda.is_available() else model

        dataset = self.__dataset_config.load()
        dataset = dataset.map(
            lambda examples: self.__tokenize_and_align_labels(tokenizer, examples),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)

        training_args = TrainingArguments(
            output_dir=str(self.__workspace_config.checkpoint_dir),
            logging_dir=str(self.__workspace_config.logging_dir),
            report_to=["tensorboard"],
        )
        training_args = self.__training_config.extend(training_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(self.__workspace_config.model_dir))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)

        if self.__training_config.push_to_hub:
            model.push_to_hub(self.__model_name)
            tokenizer.push_to_hub(self.__model_name)

    def __tokenize_and_align_labels(
        self, tokenizer: RobertaTokenizerFast, examples: dict
    ) -> dict:
        tokenized_inputs = tokenizer(
            examples["words"],
            truncation=True,
            max_length=self.__context_length,
            padding="max_length",
            is_split_into_words=True,
        )
        labels = []

        for i, label in enumerate(examples["pos"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    try:
                        # ちゃんとやりたい。
                        label_ids.append(label_names.index(label[word_idx]))
                    except ValueError:
                        label_ids.append(-100)
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # https://huggingface.co/docs/transformers/en/tasks/token_classification#evaluate
    def compute_metrics(p: EvalPrediction) -> dict:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]  # noqa: E741
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]  # noqa: E741
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
