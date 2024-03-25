FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
WORKDIR /trainer

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN apt update \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . ./
RUN pip install --no-cache-dir --no-deps .

RUN python -m ainu_lm_trainer.app.main cache

ENTRYPOINT ["python", "-m", "ainu_lm_trainer.app.main"]
