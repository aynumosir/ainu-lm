from ..config import (
    DatasetsConfigWithHuggingFaceHub,
    FineTuningConfig,
    TrainingConfig,
    WorkspaceConfig,
)
from ..services import (
    ByteLevelBpeTokenizerTrainer,
    Gpt2Trainer,
    Mt5AffixTrainer,
    Mt5GecTrainer,
    Mt5Trainer,
    RobertaPosTrainer,
    RobertaTrainer,
)
from .argument_parser import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    dataset_config = DatasetsConfigWithHuggingFaceHub(
        name=args.dataset_name,
        split=args.dataset_split,
        revision=args.dataset_revision,
    )
    fine_tuning_config = FineTuningConfig(
        tokenizer=args.base_tokenizer,
        model=args.base_model,
    )
    training_config = TrainingConfig(
        num_train_epochs=args.num_train_epochs,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        push_to_hub=args.push_to_hub,
    )
    workspace_config = WorkspaceConfig(
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        logging_dir=args.logging_dir,
    )

    if args.task == "roberta":
        roberta_trainer = RobertaTrainer(
            dataset_config=dataset_config,
            fine_tuning_config=fine_tuning_config,
            training_config=training_config,
            workspace_config=workspace_config,
        )
        roberta_trainer.train()

    if args.task == "roberta-pos":
        roberta_pos_trainer = RobertaPosTrainer(
            dataset_config=dataset_config,
            fine_tuning_config=fine_tuning_config,
            training_config=training_config,
            workspace_config=workspace_config,
        )
        roberta_pos_trainer.train()

    if args.task == "mt5":
        mt5_trainer = Mt5Trainer(
            dataset_config=dataset_config,
            fine_tuning_config=fine_tuning_config,
            training_config=training_config,
            workspace_config=workspace_config,
        )
        mt5_trainer.train()

    if args.task == "mt5-affix":
        mt5_affix_trainer = Mt5AffixTrainer(
            dataset_config=dataset_config,
            fine_tuning_config=fine_tuning_config,
            training_config=training_config,
            workspace_config=workspace_config,
        )
        mt5_affix_trainer.train()

    if args.task == "mt5-gec":
        mt5_gec_trainer = Mt5GecTrainer(
            # dataset_config=dataset_config,
            fine_tuning_config=fine_tuning_config,
            training_config=training_config,
            workspace_config=workspace_config,
        )
        mt5_gec_trainer.train()

    if args.task == "gpt2":
        gpt2_trainer = Gpt2Trainer(
            dataset_config=dataset_config,
            fine_tuning_config=fine_tuning_config,
            training_config=training_config,
            workspace_config=workspace_config,
        )
        gpt2_trainer.train()

    if args.task == "byte-level-bpe":
        bpe_trainer = ByteLevelBpeTokenizerTrainer(
            dataset_config=dataset_config,
            workspace_config=workspace_config,
        )
        bpe_trainer.train()
