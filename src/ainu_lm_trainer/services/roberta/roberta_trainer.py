import torch
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
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


class RobertaTrainer:
    __context_length = 128
    __model_name = "aynumosir/roberta-base-ainu"

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
            str(self.__fine_tuning_config.tokenizer)
        )

        config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

        model = RobertaForMaskedLM(config)
        model = model.to("cuda") if torch.cuda.is_available() else model

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        dataset = self.__dataset_config.load()
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=self.__context_length,
                padding="max_length",
                return_tensors="pt",
            ),
            batched=True,
        )
        dataset_dict = dataset.train_test_split(test_size=0.1)

        training_args = TrainingArguments(
            output_dir=str(self.__workspace_config.checkpoint_dir),
            evaluation_strategy="epoch",
            logging_dir=str(self.__workspace_config.logging_dir),
            report_to=["tensorboard"],
        )
        training_args = self.__training_config.extend(training_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
        )

        trainer.train()
        trainer.save_model(str(self.__workspace_config.model_dir))

        metrics = trainer.evaluate(eval_dataset=dataset_dict["test"])
        trainer.save_metrics("all", metrics)

        if self.__training_config.push_to_hub:
            model.push_to_hub(self.__model_name)
            tokenizer.push_to_hub(self.__model_name)
