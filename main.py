'''
In this file we are going to load the data and other ML pipeline techniques
which are needed
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from missing_value import random_sample
from variable_transform import VT_OUT
from feature_selection import FeatureSelection
from balance_data import BALANCING_DATA
from all_model import common
from sklearn.preprocessing import StandardScaler
import pickle

class HEART_DISEASE_PREDICT:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info('Data loaded succesfully')

            logger.info(f'Total Rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'Checking Null Values : {self.df.isnull().sum()}')

            self.X = self.df.iloc[:, :-1]  # independent
            self.y = self.df.iloc[:, -1].astype(int)  # dependent

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            missing_value = setup_logging('missing_value')
            missing_value.info(f'total rows in training data: {self.X_train.shape}')
            missing_value.info(f'total rows in testing data: {self.X_test.shape}')
            missing_value.info(f"Before : {self.X_train.columns}")
            missing_value.info(f"Before : {self.X_test.columns}")
            missing_value.info(f"Before : {self.X_train.isnull().sum()}")
            missing_value.info(f"Before : {self.X_test.isnull().sum()}")
            self.X_train, self.X_test = random_sample.random_sample_imputation_technique(self.X_train, self.X_test)
            missing_value.info(f"After : {self.X_train.columns}")
            missing_value.info(f"After : {self.X_test.columns}")
            missing_value.info(f"After : {self.X_train.isnull().sum()}")
            missing_value.info(f"After : {self.X_test.isnull().sum()}")
            missing_value.info(f'total rows in training data: {self.X_train.shape}')
            missing_value.info(f'total rows in testing data: {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def vt_out(self):
        try:
            vt_out = setup_logging('variable_transform')
            vt_out.info('Variable Transformation and Outlier Detection')
            for i in self.X_train.columns:
                vt_out.info(f'{self.X_train[i].dtype}')
            vt_out.info(f'{self.X_train.columns}')
            vt_out.info(f'{self.X_test.columns}')

            self.X_train, self.X_test = VT_OUT.variable_transformation_outlier(self.X_train, self.X_test)

            vt_out.info(f'{self.X_train.columns} --> {self.X_train.shape}')
            vt_out.info(f'{self.X_test.columns} --> {self.X_test.shape}')

            vt_out.info('Variable Transformation Completed')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def fs(self):
        try:
            fs = setup_logging('feature_selection')
            fs.info(f"Before : {self.X_train.columns}->{self.X_train.shape}")
            fs.info(f"Before : {self.X_test.columns}->{self.X_test.shape}")
            self.X_train, self.X_test = FeatureSelection.run(self.X_train, self.X_test,
                                                                           self.y_train)
            fs.info(f"After : {self.X_train.columns}->{self.X_train.shape}")
            fs.info(f"After : {self.X_test.columns}->{self.X_test.shape}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def balance(self):
        try:
            balance = setup_logging('balance_data')
            balance.info(f"Before - Class Distribution: {self.y_train.value_counts().to_dict()}")
            self.X_train, self.y_train = BALANCING_DATA.balance_data(self.X_train, self.y_train)
            balance.info(f"After - Class Distribution: {pd.Series(self.y_train).value_counts().to_dict()}")
            balance.info(f"Balanced X_train shape: {self.X_train.shape}")
            balance.info(f"Balanced y_train shape: {self.y_train.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_scaling(self):
        try:
            logger.info(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            logger.info(f"Before scaling (train):\n{self.X_train.head()}")
            logger.info(f"Before scaling (test):\n{self.X_test.head()}")

            # Columns you want to scale
            scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            existing_cols = [col for col in scale_cols if col in self.X_train.columns]

            if existing_cols:
                scaler = StandardScaler()
                scaler.fit(self.X_train[existing_cols])

                # Transform both train and test
                self.X_train[existing_cols] = scaler.transform(self.X_train[existing_cols])
                self.X_test[existing_cols] = scaler.transform(self.X_test[existing_cols])

                # Save scaler for inference
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)

                logger.info(f"Scaled columns: {existing_cols}")
                logger.info(f"After scaling (train):\n{self.X_train.head()}")
                logger.info(f"After scaling (test):\n{self.X_test.head()}")
            else:
                logger.warning("No columns available for scaling.")

            return self.X_train, self.X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def models(self):
        try:
            logger.info(f'Training Started')
            common(self.X_train, self.y_train, self.X_test, self.y_test)
            logger.info(f'Training Completed')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

if __name__ == "__main__":
    try:

        obj = HEART_DISEASE_PREDICT('D:\\Heart_disease_prediction\\pythonProject\\heart (1).csv')
        obj.missing_values()
        obj.vt_out()
        obj.fs()
        obj.balance()
        obj.data_scaling()
        obj.models()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')