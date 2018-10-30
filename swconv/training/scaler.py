from swconv.training import BaseFactory
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, \
                                Normalizer, Imputer, Binarizer, PolynomialFeatures


class DataScaler(BaseFactory):
    def __init__(self, dataset, logger_instance=None):
        """
        :param dataset Dataset 객체
        """
        BaseFactory.__init__(self, dataset, logger_instance=logger_instance)
        self.TAG = "Scaler"

    def get(self, scaler=None):
        """
        주어진 Scaler를 사용하여 값을 조정합니다.
        :returns scaler, transformed_dataset
        """

        # 사용 가능한 Scaler의 Dictionary
        scaler_set = {
            "standard": self.__standard_scale,
            "robust": self.__robust_scale,
            "minmax": self.__minmax_scale,
            "normalizer": self.__normalize_scale
        }

        if scaler is None:
            # Scale을 하지 않고 그대로 내보냄
            return self.dataset
        elif scaler in scaler_set.keys():
            # 알맞은 Scaler에 할당하여 처리함
            return scaler_set[scaler]()
        else:
            raise ValueError("적합한 전처리 키워드가 아닙니다.")

    def __standard_scale(self):
        """표준정규분포 형태로 변환합니다."""
        self.log("StandardScaler (표준정규분포) 진행 중")
        transformed_dataset = \
            StandardScaler().fit(self.dataset.data).transform(self.dataset.data)
        self.dataset.data = transformed_dataset
        return self.dataset

    def __robust_scale(self):
        """중간값 및 4분위값을 기준으로 분포를 변환합니다."""
        self.log("RobustScaler (중간값, 4분위값 분포) 진행 중")
        transformed_dataset = \
            RobustScaler().fit(self.dataset.data).transform(self.dataset.data)
        self.dataset.data = transformed_dataset
        return self.dataset

    def __minmax_scale(self):
        """모든 값을 0과 1 사이로 가둡니다."""
        self.log("MinMaxScaler (0과 1사이로 값 조절) 진행 중")
        transformed_dataset = \
            MinMaxScaler().fit(self.dataset.data).transform(self.dataset.data)
        self.dataset.data = transformed_dataset
        return self.dataset

    def __normalize_scale(self):
        """모든 Feature Vector의 Euclidean Length를 1로 맞춥니다."""
        self.log("Normalizer (Euclidean Length) 진행 중")
        transformed_dataset = \
            Normalizer().fit(self.dataset.data).transform(self.dataset.data)
        self.dataset.data = transformed_dataset
        return self.dataset