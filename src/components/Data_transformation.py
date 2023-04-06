import sys, os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation():

    def __init__(self):
        self.transformation_config = DataTransformationConfig

    def __get_data_transformer_obj(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_cols = ['writing_score','reading_score']
            categporical_cols = ['gender','race_ethnicity','parental_level_of_education',
                                 'lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
            ]
            )
            logging.info("Numerical columns imputation & scaling completed")
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse=False)),
                    ("scaler",StandardScaler(with_mean=False))
            ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categporical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def intitiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train & test data completed")

            pre_processing_obj = self.__get_data_transformer_obj()

            target_col_name = "math_score"
            numerical_cols = ['writing_score', 'reading_score']
            categporical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education',
                                 'lunch', 'test_preparation_course']

            input_feature_train_df = train_df.drop(columns = [target_col_name], axis= 1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns = [target_col_name], axis= 1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = pre_processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = pre_processing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocesing object")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = pre_processing_obj
            )

            logging.info("Data Transformation Complete")

            return (train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)








