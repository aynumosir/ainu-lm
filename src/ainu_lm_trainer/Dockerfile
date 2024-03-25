FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
WORKDIR /trainer

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN apt update \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /trainer

# https://stackoverflow.com/a/74240630
RUN pip install -q pip-tools \
 && pip-compile -q -o requirements.txt pyproject.toml \
 && pip install -q --no-cache-dir -r requirements.txt

COPY . /trainer
RUN pip install -q --no-cache-dir --no-deps .

RUN python -m ainu_lm_trainer.app.main cache

ENTRYPOINT ["python", "-m", "ainu_lm_trainer.app.main"]
