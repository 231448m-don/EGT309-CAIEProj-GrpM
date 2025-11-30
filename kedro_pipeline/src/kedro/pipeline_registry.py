from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.models.pipeline import create_pipeline as models_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "models": models_pipeline(),
        "__default__": models_pipeline(),
    }
