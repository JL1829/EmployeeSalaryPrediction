"""
Author: Johnny Lu
Date: 6th/July/2020   
Copyright: Johnny Lu, 2020
email: joh@johdev.com
website: https://johdev.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#################################
# self packages
from src.preprocessing import DataSetGenerator
from src.features.build_features import FeatureGenerator
from src.models.predict_model import ModelGenerator


def print_summary(models, data):
    '''prints summary of models, best model, and feature importance'''
    print('\nModel Summaries:\n')
    for model in models.mean_mse:
        print('\n', model, '- MSE:', models.mean_mse[model])
    print('\nBest Model:\n', models.best_model)
    print('\nMSE of Best Model\n', models.mean_mse[models.best_model])
    print('\nFeature Importances\n', models.get_feature_importance(models.best_model, data.feature_cols))

    feature_importances = models.get_feature_importance(models.best_model, data.feature_cols)
    feature_importances.plot.bar()
    plt.show()    

def main(train_feature_file:str, 
         train_target_file:str, test_file:str, 
         cat_cols:list, num_cols:list, target_col:str, id_col:str):
    
    print("Now start accessing data and preprocess it. \n")
    data = DataSetGenerator(train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, id_col)
    
    feature_engineering = True

    if feature_engineering:
        FeatureGenerator(data).add_group_stats()
    
    models = ModelGenerator(models=[], data=data)

    models.add_model(LinearRegression())
    models.add_model(RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=15, min_samples_split=80,
                                           max_features=8))
    models.add_model(GradientBoostingRegressor(n_estimators=100, max_depth=7))
    models.add_model(XGBRegressor())
    models.add_model(LGBMRegressor())
    models.cross_validate()
    models.select_best_model()

    models.best_model_fit(data.train_df[data.feature_cols], data.train_df[data.target_col])
    models.best_model_predict(data.test_df[data.feature_cols])
    
    print_summary(models, data)


if __name__ == "__main__":
    # define input files
    train_feature_file = 'data/raw/train_features.csv'
    train_target_file = 'data/raw/train_salaries.csv'
    test_file = 'data/raw/test_features.csv'

    # define variables
    cat_cols = ['companyId', 'jobType', 'degree', 'major', 'industry']
    num_cols = ['yearsExperience', 'milesFromMetropolis']
    target_col = 'salary'
    id_col = 'jobId'

    main(train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, id_col)
