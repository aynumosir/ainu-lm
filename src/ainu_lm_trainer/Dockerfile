FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt update

WORKDIR /trainer

COPY . /trainer

RUN pip install -e .

RUN python3 -m src.dazai_bert.cli.main cache

ENTRYPOINT ["python3", "-m", "src.ainu_lm_trainer.app.main"]
