from sklearn.decomposition import PCA

from swconv.training import BaseFactory


class DataPCA(BaseFactory):
    def __init__(self, dataset, logger_instance=None):
        """
        :param dataset Dataset 객체
        """
        BaseFactory.__init__(self, dataset, logger_instance=logger_instance)
        self.TAG = "PCA"

    def get(self):
        """
        주어진 Dataset에 PCA 전처리를 수행합니다.
        :return dataset 전처리된 데이터
        """
        self.log("PCA 전처리 진행 중")
        transformed_data = PCA(n_components=10).fit(self.dataset.data).transform(self.dataset.data)
        self.dataset.data = transformed_data
        return self.dataset