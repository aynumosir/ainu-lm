import os

PROJECT_ID = os.getenv("PROJECT_ID", "neetlab")
REGION = os.getenv("REGION", "us-central1")
TRAIN_IMAGE_URI = os.getenv(
    "TRAIN_IMAGE_URI", "us-central1-docker.pkg.dev/neetlab/docker/ainu-lm-trainer"
)
PIPELINE_ROOT = "gs://ainu-lm/pipeline_root"
PIPELINE_STAGING = "gs://ainu-lm/staging"
