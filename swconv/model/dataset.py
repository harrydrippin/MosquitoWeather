import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, df, logger_instance=None):
        """
        DataFrame을 통해서 Dataset을 생성합니다.
        :param df Pandas DataFrame
        """
        self.log = logger_instance
        self._raw_df = df.copy()
        self.dataframe = df.copy()
        self.labels = np.array(self.dataframe['actual'])
        self.dataframe.drop('actual', axis=1, inplace=True)
        self.feature_names = list(self.dataframe.columns)
        self.data = self.dataframe

    def _log(self, log_string):
        """
        Logger의 상태에 따라 기록을 남깁니다.
        """
        if self.log is None: return
        else: self.log.debug(f"Dataset: {log_string}")

    def data_split(self):
        """
        데이터를 목적에 맞게 나눕니다.
        """
        self.train_data, self.test_data, self.train_labels, self.test_labels \
            = train_test_split(self.data, self.labels, test_size=0.25, random_state=42)
        self._log(f'Train Data Shape: {self.train_data.shape}')
        self._log(f'Train Label Shape: {self.train_labels.shape}')
        self._log(f'Test Data Shape: {self.test_data.shape}')
        self._log(f'Test Label Shape: {self.test_labels.shape}')