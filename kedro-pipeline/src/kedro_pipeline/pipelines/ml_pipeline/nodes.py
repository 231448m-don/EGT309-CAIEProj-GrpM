from .ml_models.logistic_regression import train_logistic_regression
from .ml_models.random_forest import train_random_forest
from .ml_models.xgboost_model import train_xgboost

def lr_node(input_data, target_col):
    return train_logistic_regression(input_data, target_col)

def rf_node(input_data, target_col):
    return train_random_forest(input_data, target_col)

def xgb_node(input_data, target_col):
    return train_xgboost(input_data, target_col)
