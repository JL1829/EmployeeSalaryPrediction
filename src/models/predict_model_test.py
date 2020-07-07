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


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


class ModelGenerator(object):
    """Doc Strings goes here"""

    def __init__(self, models, data):
        self.models = list(models)
        self.data = data
        self.best_model = None
        self.predictions = None
        self.mean_mse = {}
    
    def add_model(self, model):
        self.models.append(model)

    def cross_validate(self, k=5, num_procs=4):
        '''cross validate models using given data'''
        feature_df = self.data.train_df[self.data.feature_cols]
        target_df = self.data.train_df[self.data.target_col]
        for model in self.models:
            neg_mse = cross_val_score(model, feature_df, target_df, cv=k, n_jobs=num_procs, scoring='neg_mean_squared_error')
            self.mean_mse[model] = -1.0 * np.mean(neg_mse)