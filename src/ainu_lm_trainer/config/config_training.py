from dataclasses import dataclass
from typing import Optional

from transformers import TrainingArguments


@dataclass
class TrainingConfig:
    num_train_epochs: Optional[int] = None

    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None

    weight_decay: Optional[float] = None
    learning_rate: Optional[float] = None
    warmup_ratio: Optional[float] = None

    hub_model_id: Optional[str] = None
    push_to_hub: Optional[bool] = False

    def extend(self, args: TrainingArguments) -> TrainingArguments:
        for key, value in self.__dict__.items():
            if value is not None:
                setattr(args, key, value)
        return args
