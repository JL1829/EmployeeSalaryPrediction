###################################################
#                                                 #
#                                                 #
# Author: Johnny Lu                               #
# Date: 6th/July/2020                             #    
# Copyright: Johnny Lu, 2020                      #
# email: johnnylou89@icloud.com                   #
# website: https://johdev.com                     #
#                                                 #
#                                                 #
###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#################################
# self packages
from src import preprocessing
from src.features import build_features
from src.models import predict_model

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

    data = preprocessing.DataSetGenerator(train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, id_col)

    feature_engineering = True

    if feature_engineering:
        feature_generator = build_features.FeatureGenerator(data)
        feature_generator.add_group_stats()
    
    models = predict_model.ModelGenerator(models=[], data=data)

    models.add_model(LinearRegression())
    models.add_model(RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=15, min_samples_split=80,
                                           max_features=8))
    models.add_model(GradientBoostingRegressor(n_estimators=100, max_depth=7, loss='ls'))
    models.add_model(XGBRegressor())
    models.add_model(LGBMRegressor())

    models.select_best_model()

    models.best_model_fit(data.train_df[data.feature_cols], data.train_df[data.target_col])
    models.best_model_predict(data.test_df[data.feature_cols])

def print_summary(models):
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
