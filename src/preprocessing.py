"""
Author: Johnny Lu
Date: 6th/July/2020   
Copyright: Johnny Lu, 2020
email: joh@johdev.com
website: https://johdev.com
"""


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


class DataSetGenerator(object):
    """A Data Set Generator/Processor to perform the basic DataFrame operation:
    * Create Pandas DataFrame from files or URLs
    * Merge DataFrames if Features and Target label are sepreated file.
    * Catagorical Feature encoding
    * Data Cleaning (remove duplicate, remove outliers, imputation(planed))
    
    
    Parameter
    ------------
    - train_feature_file: Type: str, file path or URLs
    - train_target_file: Type: str, file path or URLs
    - test_file: Type: str, file path or URLs
    - cat_cols: Type: List, name list of categorical columns
    - num_cols: Type: List, name list of numerical columns
    - target_col: Type: str or List, the name of Target Label column
    - index_col: Type: str, the name of index column
    

    Attributes:
    ------------
    train_df: 
        A well preprocessed Pandas DataFrame contain features and label columns
    
    test_df:
        A well label encoded Pandas DataFrame contain feature columns only

    label_encoders: 
        dict, key: column name, value: LabelEncoder()


    Method
    ------------
    - label_encode_df(self, df, col):
        Label encoder for df and col, it will search whether the Label Encoder instance 
        already exist, it not, create Label Encoder instance for particular column
        and perform label encoding. 
    
    
    Example
    ------------
    >>> data = DataSetGenerator(train_feature_file, train_target_file, 
                            test_file, cat_cols, num_cols, target_col, id_col)
    
    >>> data.train_df
         ### DataFrame ###
    
    >>> data.test_df
         ### DataFrame### 
    
    >>> data.feature_cols
         ### feature columns shows here
    
    >>> data.label_encoders
         ### print out label encoder dict"""
    

    def __init__(self, 
                 train_feature_file, 
                 train_target_file, 
                 test_file, 
                 cat_cols, num_cols, target_col, index_col):
        """ Create train and test DataFrame"""
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.feature_cols = cat_cols + num_cols
        self.target_col = target_col
        self.index_col = index_col
        self.label_encoders = {}
        self.train_df = self._create_train_df(train_feature_file, train_target_file)
        self.test_df = self._create_test_df(test_file)
    
    def _create_train_df(self, train_feature_file, train_target_file,
                         preprocess=True, label_encode=True):
        """ loads and merges training data features and targets
        preprosses data, and label encode data"""
        train_feature_df = self._loadData(train_feature_file)
        train_target_df = self._loadData(train_target_file)
        train_df = self._merge_dfs(train_feature_df, train_target_df)

        if preprocess:
            train_df = self._cleanData(train_df)
            train_df = self._shuffleData(train_df)
        
        if label_encode:
            self.label_encode_df(train_df, self.cat_cols)
        
        return train_df
  
    def _create_test_df(self, test_file, label_encode=True):
        """loads and label encodes test data"""
        test_df = self._loadData(test_file)
        if label_encode:
            self.label_encode_df(test_df, self.cat_cols)
        
        return test_df
    
    def _loadData(self, file):
        return pd.read_csv(file)
    
    def _merge_dfs(self, df1, df2, key=None, left_index=False, right_index=False):
        return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)
    
    def _cleanData(self, df):
        """remove rows that contain salary <= 8.5 or duplicated"""
        df = df.drop_duplicates()
        df = df[df['salary'] > 8.5]

        return df

    def _shuffleData(self, df):
        return shuffle(df).reset_index()
    
    def label_encode_df(self, df, cols):
        """creates one label encoder for each column in the data object instance"""
        for col in cols:
            if col in self.label_encoders:
                # if label encoder already exits for col, use it
                self._label_encode(df, col, self.label_encoders[col])
            else:
                self._label_encode(df, col)
   
    def _label_encode(self, df, col, le=None):
        """label encodes data"""
        if le:
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            self.label_encoders[col] = le
