"""
Author: Johnny Lu
Date: 6th/July/2020   
Copyright: Johnny Lu, 2020
email: joh@johdev.com
website: https://johdev.com
"""

import numpy as np
import pandas as pd

class FeatureGenerator(object):
    """A feature engineering generator to create additional feature to DataFrame
    
    Group by each label encoded column, compute the:
        * mean
        * max
        * min
        * Standard Deviation
        * median
        of label column
    
    Parameter
    -------------
    data: object, a dataset object created by `preprocessing.py`
    

    Method
    -------------
    add_group_stats(self):
        group by each label encoded column, and compute the Grouped Statitics. 
        fill any NaN value with 0

        Return: grouped by statitics DataFrame merged with original DataFrame.
    

    Example
    -------------
    >>> feature_engineering = True
    >>> if feature_engineering:
            FeatureGenerator(data).add_group_stats()
    """

    def __init__(self, data:object):
        """initializes class and creates groupby object for data"""
        self.data = data
        self.cat_cols = data.cat_cols
        self.groups = data.train_df.groupby(self.cat_cols)
    
    def add_group_stats(self):
        """adds group statistics to data stored in data object"""
        group_stats_df = self._get_group_stats()
        group_stats_df.reset_index(inplace=True)

        # merge derived columns to original df
        self.data.train_df = self._merge_new_cols(self.data.train_df, group_stats_df, self.cat_cols, fillna=True)
        self.data.test_df = self._merge_new_cols(self.data.test_df, group_stats_df, self.cat_cols, fillna=True)

        # update column list
        group_stats_cols = ['group_mean_salary', 'group_max_salary', 'group_min_salary', 'group_std_salary', 'group_median_salary']
        self._extend_col_lists(self.data, cat_cols=group_stats_cols)
    
    def _get_group_stats(self):
        """calculate group statistics"""
        target_col = self.data.target_col
        group_stats_df = pd.DataFrame({'group_mean_salary': self.groups[target_col].mean()})
        group_stats_df['group_max_salary'] = self.groups[target_col].max()
        group_stats_df['group_min_salary'] = self.groups[target_col].min()
        group_stats_df['group_std_salary'] = self.groups[target_col].std()
        group_stats_df['group_median_salary'] = self.groups[target_col].median()

        return group_stats_df
    
    def _merge_new_cols(self, df, new_cols_df, keys, fillna=False):
        """Merges engineered features with original df"""
        DataFrame = pd.merge(df, new_cols_df, on=keys, how='left')
        if fillna:
            DataFrame.fillna(0, inplace=True)
        
        return DataFrame
    
    def _extend_col_lists(self, data, cat_cols=[], num_cols=[]):
        """addes engineered features cols to data cols lists"""
        data.num_cols.extend(num_cols)
        data.cat_cols.extend(cat_cols)
        data.feature_cols.extend(num_cols + cat_cols)
