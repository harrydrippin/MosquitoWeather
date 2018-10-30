from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swconv.training import BaseFactory
import pickle
import time

class DataModel:
    def __init__(self, dataset, logger_instance=None):
        if dataset is None:
            raise ValueError("Dataset이 없습니다.")
        if dataset.train_data is None:
            raise ValueError("Data Split이 이루어지지 않은 Dataset입니다.")
        self.dataset = dataset
        self.log = logger_instance
        self.model = None
        self.model_type = None

    def _log(self, log_string):
        """
        Logger의 상태에 따라 기록을 남깁니다.
        """
        if self.log is None: return
        else: self.log.debug(f"DataModel: {log_string}")

    def run(self, model_keyword):
        """
        주어진 모델 키워드로 학습을 진행합니다.
        """
        self.model_type = model_keyword
        if model_keyword == "random_forest":
            self.model = RandomForestRegressor(n_estimators=1000, random_state=42)
            self._log("Random Forest를 학습합니다.")
        elif model_keyword == "svm":
            self.model = SVR(kernel="rbf", C=2.0)
            self._log("SVM Regression을 학습합니다.")
        elif model_keyword == "decision_tree":
            self.model = tree.DecisionTreeRegressor(random_state=32)
            self._log("Decision Tree Regression을 학습합니다.")
        else:
            self.model_type = None
            raise ValueError(f"{model_keyword}은(는) 존재하지 않는 모델 이름입니다.")
        start_time = time.time()
        self.model.fit(self.dataset.train_data, self.dataset.train_labels)
        end_time = time.time()
        self._log("완료되었습니다.")
        self._log("소요 시간: %2.2f초" % (end_time - start_time))

    def apply_level(self, data):
        """
        예측 결과를 등급화합니다.
        :param data 예측 결과
        """
        result = list()

        for i in data:
            if i <= 10:
                result.append(1)
            elif i <= 20:
                result.append(2)
            elif i <= 100:
                result.append(3)
            else:
                result.append(4)
        return np.array(result)

    def test(self, level=False, fig_save=False, key=None):
        """
        보유한 모델로 Test를 실시합니다.
        :param level 값의 등급화 여부
        """
        # TODO(@harrydrippin): 평가 알고리즘 수정
        self._log("Testset으로 예측을 진행합니다.")
        predictions = self.model.predict(self.dataset.test_data)
        test_dataset = self.dataset.test_labels

        if level:
            self._log("값의 등급화를 수행합니다.")
            predictions = self.apply_level(predictions)
            test_dataset = self.apply_level(test_dataset)

        # 잔차 분석: 평균 제곱 오차
        errors = abs(predictions - test_dataset)
        mse = mean_squared_error(test_dataset, predictions)
        self._log(f"평균 제곱 오차: {mse}")

        if fig_save:
            plt.figure()
            plt.subplot(121)
            plt.plot(predictions[0:20], label="Prediction")
            plt.plot(test_dataset[0:20], label="Answer")
            plt.legend(loc="upper right", fontsize=10)
            plt.subplot(122)
            plt.plot(errors[0:20], label="Error")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(f"./images/{key}.png")
            plt.close()

        if level:
            # Accuracy
            self._log("값의 등급화를 수행하였으므로 결정 계수를 정확도로 대체합니다.")
            accuracy = np.bincount(errors, minlength=np.max(errors))[0] / len(errors)
            self._log(f"정확도: {accuracy * 100}%")
            return mse, accuracy
        else:
            # 결정 계수
            r2 = r2_score(test_dataset, predictions)
            self._log(f"결정 계수: {r2}")
            return mse, r2


    def save_model(self, path):
        """
        주어진 Path로 모델을 Pickling해서 넣습니다.
        """
        self._log(f"Saving model from {path}...")
        if self.model is None:
            raise ValueError("객체 인스턴스에 훈련된 모델이 없습니다.")
        with open(path, 'wb') as f:
            pickle.dumps(self, f)

    @classmethod
    def load_model(cls, path):
        """
        주어진 Path에서 Pickling된 모델을 불러옵니다.
        """
        with open(path, 'rb') as f:
            datamodel = pickle.load(f)
        return datamodel