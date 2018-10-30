import pandas as pd
from config import Config
import os
import sys
import numpy as np
class DataLoader:
    def __init__(self, logger_instance=None):
        """
        DataLoader를 초기화합니다.
        :param logger_instance Logger 객체
        """
        self.log = logger_instance

    def load(self):
        """
        각 데이터를 전처리 및 병합하여 반환합니다.
        :return Pandas DataFrame
        """
        # key: data type, value: dataframe

        data_df = {}
        for data_path in Config.dataset_path_list:
            data_type = data_path.split('_')[-1].split('.')[0]
            data_df[data_type] = self._load_datasets(data_path,data_type)

        return self._refine(data_df)

    def _log(self, log_string):
        """
        Logger의 상태에 따라 기록을 남깁니다.
        """
        if self.log is None: return
        else: self.log.debug(f"DataLoader: {log_string}")


    def _load_datasets(self, data_path, data_type):
        """
        명시된 데이터를 불러옵니다.
        :param data_type 데이터의 종류
        :return Pandas DataFrame
        """
        self._log(f"{data_type} Dataset을 읽는 중입니다...")
        if not os.path.exists(data_path):
            raise ValueError(f"데이터({data_type})를 찾을 수 없습니다.")
        
        return pd.read_csv(data_path, index_col="date")


    def _refine(self, data_csv_dic):
        """
        Data의 Index를 정리하여 병합합니다.
        :param data_mosq 모기 포집도 데이터
        :param data_weather 날씨 누적 데이터
        :return Merged Pandas DataFrame
        """

        refine_dataframe = []
        self._log(f"'weather' 데이터를 Index 기반으로 정리합니다.")
        data_csv_dic['weather']['rainfall'].fillna(np.float32(0), inplace=True)
        refine_dataframe.append(data_csv_dic['weather'])

        self._log(f"'mosq' 데이터를 Index 기반으로 정리합니다.")
        df = data_csv_dic["mosq"]
        new_df = pd.DataFrame([], columns=df.columns)

        for idx in df.index:
            index_name = idx[0:10]
            if index_name not in new_df.index:
                new_df.loc[index_name] = df.loc[idx].fillna(np.mean(df.loc[idx]))
            else:
                new_df.loc[index_name] += df.loc[idx].fillna(np.mean(df.loc[idx]))

        new_df = pd.DataFrame(new_df.mean(axis=1),columns=['actual'])
        refine_dataframe.append(new_df)

        self._log("Dataset을 합칩니다.")
        all_df = pd.concat(refine_dataframe, axis=1, sort=True)
        return all_df