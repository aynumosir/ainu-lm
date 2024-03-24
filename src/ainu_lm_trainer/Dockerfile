FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
WORKDIR /workspace

ARG HF_TOKEN
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt update
COPY . /workspace

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry export -f requirements.txt --output requirements.txt
RUN pip install -r requirements.txt

RUN python3 -m src.ainu_lm_trainer.app.main cache

ENTRYPOINT ["python3", "-m", "src.ainu_lm_trainer.app.main"]
