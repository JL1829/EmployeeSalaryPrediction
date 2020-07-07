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

    def __init__(self, models, data):#, default_num_iters=10, verbose_lvl=0):
        '''initializes model list and dicts'''
        self.models = list(models)
        self.data = data
        self.best_model = None
        self.predictions = None
        self.mean_mse = {}
        #self.default_num_iters = default_num_iters
        #self.verbose_lvl = verbose_lvl
        
    def add_model(self, model):
        self.models.append(model)

    def cross_validate(self, k=5, num_procs=-1):
        '''cross validate models using given data'''
        feature_df = self.data.train_df[self.data.feature_cols]
        target_df = self.data.train_df[self.data.target_col]
        for model in self.models:
            neg_mse = cross_val_score(model, feature_df, target_df, cv=k, n_jobs=num_procs, scoring='neg_mean_squared_error')
            self.mean_mse[model] = -1.0 * np.mean(neg_mse)
    
    def select_best_model(self):
        '''select model with lowest mse'''
        self.best_model = min(self.mean_mse, key=self.mean_mse.get)
        
    def best_model_fit(self, features, targets):
        '''fits best model'''
        self.best_model.fit(features, targets)
    
    def best_model_predict(self, features):
        '''scores features using best model'''
        self.predictions = self.best_model.predict(features)
        
    def save_results(self):
        pass
    
    @staticmethod
    def get_feature_importance(model, cols):
        '''retrieves and sorts feature importances'''
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = pd.DataFrame({'feature':cols, 'importance':importances})
            feature_importances.sort_values(by='importance', ascending=False, inplace=True)
            #set index to 'feature'
            feature_importances.set_index('feature', inplace=True, drop=True)
            return feature_importances
        else:
            #some models don't have feature_importances_
            return "Feature importances do not exist for given model"
            