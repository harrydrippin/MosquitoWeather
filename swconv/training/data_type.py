from swconv.training import BaseFactory
from swconv.model.dataset import Dataset
import numpy as np
import math
import sys
class DataFormatter(BaseFactory):
    def __init__(self, dataset, logger_instance=None):
        BaseFactory.__init__(self, dataset, logger_instance=logger_instance)
        self.TAG = "Formatter"

    def get(self, number=1):
        """
        Data를 주어진 안에 따라 정렬합니다.
        :param number 데이터의 안 번호
        :return dataset 전처리된 데이터
        """
        if number == 1:
            return self.__datatype_first()
        elif number == 2:
            return self.__datatype_second()
        elif number == 3:
            return self.__datatype_third()
        elif number == 4:
            return self.__datatype_fourth()
        else:
            raise ValueError("정상적인 번호가 아닙니다. 1안부터 4안까지가 존재합니다.")
    
    def __datatype_first(self):
        """
        1안: 특정일 + 같은 날 모기 포집
        """
        self.log("1안: 특정일 + 같은 날 모기 포집")
        new_df = self.dataset.copy()
        return Dataset(new_df.dropna(axis=0), logger_instance=self.logger_instance)

    def __datatype_second(self):
        """
        2안: 특정일 + 다음 날 모기 포집
        """
        self.log("2안: 특정일 + 다음 날 모기 포집")
        new_df = self.dataset.copy()
        new_df['actual'] = np.concatenate((new_df['actual'].values[1:], [np.nan]))
        return Dataset(new_df.dropna(axis=0), logger_instance=self.logger_instance)

    def __datatype_third(self):
        """
        3안: 1안 + X에 전날 포집 모기 추가
        """
        self.log("3안: 1안 + X에 전날 포집 모기 추가")
        new_df = self.dataset.copy()
        new_df['pre_actual'] = np.insert(new_df['actual'].values,0,math.nan)[:-1]
        return Dataset(new_df.dropna(axis=0), logger_instance=self.logger_instance)

    def __datatype_fourth(self):
        """
        4안: 2안 + X에 당일 포집 모기 추가
        """
        self.log("4안: 2안 + X에 당일 포집 모기 추가")
        new_df = self.dataset.copy()
        new_df['pre_actual'] = new_df['actual']
        new_df['actual'] = new_df['actual'] = np.concatenate((new_df['actual'].values[1:], [np.nan]))
        return Dataset(new_df.dropna(axis=0), logger_instance=self.logger_instance)