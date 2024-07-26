import os
from argparse import ArgumentParser, Namespace

from ...config import (
    DatasetsConfigWithHuggingFaceHub,
    FineTuningConfig,
    MtExperimentsConfig,
    PronounType,
    TaskPrefixType,
    TrainingConfig,
    WorkspaceConfig,
)
from ...trainers import fine_tuning, pretraining
from ...utils import get_path_from_uri, get_path_str_from_uri

# fmt: off
# Dataset arguments
dataset_parser = ArgumentParser(add_help=False)
dataset_parser.add_argument("--dataset-name", type=str, required=True)
dataset_parser.add_argument("--dataset-revision", type=str)

# Training arguments
training_parser = ArgumentParser(add_help=False)
training_parser.add_argument("--per-device-train-batch-size", type=int)
training_parser.add_argument("--per-device-eval-batch-size", type=int)
training_parser.add_argument("--num-train-epochs", type=int)
training_parser.add_argument("--weight-decay", type=float)
training_parser.add_argument("--learning-rate", type=float)
training_parser.add_argument("--warmup-ratio", type=float)
training_parser.add_argument("--gradient-accumulation-steps", type=int)
training_parser.add_argument("--hub-model-id", type=str, required=True)
training_parser.add_argument("--push-to-hub", type=str, choices=["yes", "no"], default="no")

# Workspace arguments
workspace_parser = ArgumentParser(add_help=False)
workspace_parser.add_argument("--model-dir", type=get_path_from_uri, default=os.environ.get("AIP_MODEL_DIR"))
workspace_parser.add_argument("--checkpoint-dir", type=get_path_from_uri, default=os.environ.get("AIP_CHECKPOINT_DIR"))
workspace_parser.add_argument("--logging-dir", type=get_path_from_uri, default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"))

# Fine-tuning arguments
fine_tuning_parser = ArgumentParser(add_help=False)
fine_tuning_parser.add_argument("--base-model", type=get_path_str_from_uri, required=True)
fine_tuning_parser.add_argument("--base-tokenizer", type=get_path_str_from_uri)
# fmt: on


def add_parser(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="task")

    common = [
        dataset_parser,
        training_parser,
        workspace_parser,
    ]

    subparsers.add_parser("pos-tagging", parents=[*common, fine_tuning_parser])

    mt_parser = subparsers.add_parser("mt", parents=[*common, fine_tuning_parser])
    mt_parser.add_argument("--experiment-task-prefix", type=str, default="all")
    mt_parser.add_argument("--experiment-include-dialect", type=str)
    mt_parser.add_argument("--experiment-include-pronoun", type=str)
    mt_parser.add_argument("--experiment-hyperparameter-tuning", type=bool)

    roberta_parser = subparsers.add_parser("roberta", parents=common)
    roberta_parser.add_argument("--base-tokenizer", type=str)

    gpt2_parser = subparsers.add_parser("gpt2", parents=common)
    gpt2_parser.add_argument("--base-tokenizer", type=str)

    subparsers.add_parser("byte-level-bpe", parents=common)


def main(args: Namespace) -> None:
    config_dataset = DatasetsConfigWithHuggingFaceHub(
        name=args.dataset_name,
        revision=args.dataset_revision,
    )
    config_training = TrainingConfig(
        num_train_epochs=args.num_train_epochs,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub == "yes",
    )
    config_workspace = WorkspaceConfig(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        logging_dir=args.logging_dir,
    )

    if args.task == "roberta":
        pretraining.roberta.train(
            tokenizer_name=args.base_tokenizer,
            config_dataset=config_dataset,
            config_training=config_training,
            config_workspace=config_workspace,
        )

    if args.task == "pos-tagging":
        fine_tuning.pos_tagging.train(
            config_dataset=config_dataset,
            config_training=config_training,
            config_workspace=config_workspace,
            config_fine_tuning=FineTuningConfig(
                base_model=args.base_model,
                base_tokenizer=args.base_tokenizer,
            ),
        )

    if args.task == "mt":
        fine_tuning.mt.train(
            config_dataset=config_dataset,
            config_training=config_training,
            config_workspace=config_workspace,
            config_mt_experiments=MtExperimentsConfig(
                task_prefix=TaskPrefixType.from_str(args.experiment_task_prefix),
                include_dialect=args.experiment_include_dialect,
                include_pronoun=PronounType.from_str(args.experiment_include_pronoun)
                if args.experiment_include_pronoun
                else None,
                hyperparameter_tuning=args.experiment_hyperparameter_tuning,
            ),
            config_fine_tuning=FineTuningConfig(
                base_model=args.base_model,
                base_tokenizer=args.base_tokenizer,
            ),
        )

    if args.task == "gpt2":
        pretraining.gpt2.train(
            tokenizer_name=args.base_tokenizer,
            config_dataset=config_dataset,
            config_training=config_training,
            config_workspace=config_workspace,
        )

    if args.task == "byte-level-bpe":
        bpe_trainer = pretraining.ByteLevelBpeTokenizerTrainer(
            config_dataset=config_dataset,
            config_workspace=config_workspace,
        )
        bpe_trainer.train()
