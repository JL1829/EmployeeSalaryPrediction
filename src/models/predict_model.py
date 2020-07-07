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
    """Doc String goes here"""

    def __init__(self, models=[]):
        """initializes model list and dicts"""
        self.models = models
        self.best_model = None
        self.predictions = None
        self.mean_mse = {}

    def add_model(self, model):
        self.models.append(model)
    
    def cross_validate(self, data, k=5, num_proces=-1):
        """cross validate models using given data"""
        feature_df = data.train_df[data.feature_cols]
        target_df = data.train_df[data.target_col]

        for model in self.models:
            neg_mse = cross_val_score(model, feature_df, target_df, cv=k, n_jobs=num_proces, scoring='neg_mean_squared_error')
            self.mean_mse[model] = -1.0 * np.mean(neg_mse)
    
    def select_best_model(self):
        """select model with lowest mse"""
        self.best_model = min(self.mean_mse, key=self.mean_mse.get)
    
    def best_model_fit(self, features, targets):
        """fit the best model"""
        self.best_model.fit(features, targets)
    
    def best_mode_predict(self, features):
        """scores features using best model"""
        self.predictions = self.best_model.predict(features)
    
    @staticmethod
    def get_feature_importance(model, cols):
        """retrieves and sorts feature importances"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = pd.DataFrame({'feature':cols, 'importance':importances})
            feature_importances.sort_values(by='importances', ascending=False, inplace=True)

            # set index to 'feature'
            feature_importances.set_index('feature', inplace=True)
            return feature_importances
        else:
            # some model don't have feature_importances_
            return "Feature importances do not exist for given model"
    
    def print_summary(self):
        """prints summary of models, best model, and feature importances"""
        print('\nModel Summaries:\n')
        for model in self.models.mean_mse:
            print('\n', model, '- MSE:', self.models.mean_mse[model])
        
        print('\nBest Model: \n', self.models.best_model)
        print('\nMSE of Best Model: \n', self.models.mean_mse[self.models.best_model])
        print('\nFeature Importances\n', self.models.get_feature_importance(self.models.best_model, data.feature_cols))

        feature_importances = self.get_feature_importance(self.models.best_model, data.feature_cols)
        feature_importances.plot.bar()
        plt.show()
        