"""
Author: Johnny Lu
Date: 6th/July/2020   
Copyright: Johnny Lu, 2020
email: joh@johdev.com
website: https://johdev.com
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class ModelGenerator(object):
    """A General Model Generator Class to perform following opreation:
        * Add Model as a list into the Generator.
        * Perform Cross Validation process to calculate the MSE(Mean Square Error).
        * Select the model from the list which perform the best MSE result. 
        * Perform the prediction for the test data based on the best performance model. 
        
        Parameters
        ----------------
        - models: List, a list to accept Model instance from Scikit-Learn. 
        - data: Class, a data preprocessing class from `preprocessing.py`
        - best_model = None
        - predictions = None
        - mean_mse = {}
        
        Method
        ----------------
        add_model(self, model): 
           - Append the new model instance from scikit-learn into `models` list
        
        cross_validate(self, k=5, n_proces=-1):
           - Perform Cross Validation on the `models` list's model based on `data` class, 
             in default 5 folds Cross Validation, and default `-1` n_jobs
           - return the `mean_mse` dict with model name and MES
        
        select_best_model(self):
           - select the model with lowest MES value.
           - return `best_model` 
        
        best_model_fit(self, features, targets):
           - Train the best model from `best_model`
        
        best_model_predict(self, features):
           - make prediction on test set
           - return `predictions`
        
        save_results(self):
           - save the best performance model in `.pkl` file in ./models folder
        
        Static Method
        ----------------
        get_feature_importance(models, col):
           - determine whether the particular model have `feature_importances_` attribute
             if yes, print out the `feature_importances_`
        
        Examples
        ----------------
        >>> from src.preprocessing import DataSetGenerator
        >>> from src.features.build_features import FeatureGenerator
        >>> from src.models.predict_model import ModelGenerator
        >>> data = DataSetGenerator(train_feature_file, train_target_file, test_file, cat_cols, num_cols, target_col, id_col)
        >>> models = ModelGenerator(models=[], data=data)
        >>> models.add_model(LinearRegression())
        >>> models.add_model(RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=15, min_samples_split=80,
                                           max_features=8))
        >>> models.cross_validate()
        >>> models.select_best_model()
        >>> models.best_model_fit(...)
        >>> models.best_model_predict(...)
        >>> models.get_feature_importance(...)
        >>> models.save_results('model.pkl')"""

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
            