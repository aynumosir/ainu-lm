import hypertune
from transformers import TrainerCallback
from transformers.trainer_callback import (
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class HyperparameterTuningCallback(TrainerCallback):
    """
    A custom callback class that reports a metric to hypertuner
    at the end of each epoch.
    """

    def __init__(self, metric_tag: str, metric_value: str) -> None:
        super(HyperparameterTuningCallback, self).__init__()
        self.metric_tag = metric_tag
        self.metric_value = metric_value
        self.hpt = hypertune.HyperTune()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict,
    ) -> None:
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric_tag,
            metric_value=kwargs["metrics"][self.metric_value],
            global_step=state.epoch,
        )
