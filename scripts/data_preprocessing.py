#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Optional
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


class DataPreprocessing:
    def __init__(self, data: Optional[DataFrame] = None):
        self.data = data

    def read_data(self, file_path: str) -> DataFrame:
        """load data from csv file
        Args:
            file_path: str: path to the csv file
        Returns:
            DataFrame: loaded data
        """
        self.data = pd.read_csv(file_path)
        return self.data

    def remove_duplicates(self) -> DataFrame:
        """remove duplicates from the data
        Returns:
            DataFrame: data without duplicates
        """
        self.data = self.data.drop_duplicates()
        return self.data

    def label_encoding(self, column: list) -> DataFrame:
        """label encoding for categorical columns
        Args:
            column: str: column name
        Returns:
            DataFrame: data with label encoded column
        """
        le = LabelEncoder()
        for col in column:
            if self.data[col].dtype == "object":
                self.data[col] = le.fit_transform(self.data[col])

        return self.data

    def drop_columns(self, columns: list) -> DataFrame:
        """drop columns from the data
        Args:
            columns: list: list of columns to drop
        Returns:
            DataFrame: data without the dropped columns
        """
        self.data = self.data.drop(columns, axis=1)
        return self.data

    def drop_duplicates(self, column: str) -> DataFrame:
        """drop duplicates from the data
        Args:
            column: str: column name
        Returns:
            DataFrame: data without duplicates
        """
        count = self.data[column].value_counts()
        ids = count[count > 1].index
        self.data = self.data[
            ~self.data[column].isin(ids)
            | self.data.duplicated(subset=[column], keep="first")
        ]

        return self.data
