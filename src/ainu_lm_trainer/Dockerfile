FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
WORKDIR /trainer

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN apt update

COPY pyproject.toml /trainer
RUN pip install -q .

COPY . /trainer
RUN pip install -q .

RUN python -m ainu_lm_trainer.app.main cache

ENTRYPOINT ["python", "-m", "ainu_lm_trainer.app.main"]
