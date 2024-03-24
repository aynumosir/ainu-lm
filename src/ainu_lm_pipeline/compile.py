import os
from os import path

from kfp.compiler import Compiler

# PIPELINE_JSON_SPEC_PATH = path.join(
#     path.dirname(__file__), "../../pipelines/ainu_lm_pipeline.json"
# )

PIPELINE_JSON_SPEC_PATH = "./pipelines/ainu_lm_pipeline.json"

if __name__ == "__main__":
    os.makedirs(path.dirname(PIPELINE_JSON_SPEC_PATH), exist_ok=True)

    try:
        from .pipeline import ainu_lm_pipeline

        compiler = Compiler()
        compiler.compile(
            pipeline_func=ainu_lm_pipeline, package_path=PIPELINE_JSON_SPEC_PATH
        )
    except Exception as e:
        print(f"Failed to compile pipeline: {e}")
        exit(1)
