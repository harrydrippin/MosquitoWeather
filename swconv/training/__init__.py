class BaseFactory:
    def __init__(self, dataset, logger_instance=None):
        """주어진 Pandas DataFrame으로 Instance를 만듭니다."""
        self.TAG = "Factory"
        self.dataset = dataset
        self.logger_instance = logger_instance

    def log(self, log_string):
        """
        Logger의 상태에 따라 기록을 남깁니다.
        """
        if self.logger_instance is None: return
        else: self.logger_instance.debug(f"{self.TAG}: {log_string}")

    def get(self):
        """DataFrame을 조작하거나 이용하여 데이터 전처리 및 학습을 진행합니다."""
        raise TypeError("이 함수는 Override되어야 합니다.")