from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=nodes.lr_node,
                inputs=["model_input_table", "params:target_column"],
                outputs="logistic_regression_model",
                name="logistic_regression_node",
            ),
            node(
                func=nodes.rf_node,
                inputs=["model_input_table", "params:target_column"],
                outputs="random_forest_model",
                name="random_forest_node",
            ),
            node(
                func=nodes.xgb_node,
                inputs=["model_input_table", "params:target_column"],
                outputs="xgboost_model",
                name="xgboost_node",
            ),
        ]
    )
