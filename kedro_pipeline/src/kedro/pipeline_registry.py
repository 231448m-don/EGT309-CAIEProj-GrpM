from kedro_pipeline.pipelines.models import pipeline as models_pipeline

def register_pipelines():
    return {
        "__default__": models_pipeline.create_pipeline(),
        "models": models_pipeline.create_pipeline(),
    }
