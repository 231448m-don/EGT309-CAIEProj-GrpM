"""Project pipelines."""
from kedro.pipeline import Pipeline
from kedro_pipeline.pipelines import ml_pipeline  # 👈 your folder name


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    training_pipeline = ml_pipeline.create_pipeline()

    return {
        "__default__": training_pipeline,
        "ml_pipeline": training_pipeline,
    }
