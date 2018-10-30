import numpy as np
import logging
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

class Config:
    """설정 값을 보관합니다."""
    # Dataset의 Path
    dataset_path_list = [
        "./dataset/dataset_mosq.csv",
        "./dataset/dataset_weather.csv"
    ]

    # Model이 출력될 Path
    model_output = "./result"

    # Logging Constants
    LOG_LEVEL = logging.DEBUG
    LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

    # Slack Webhooks
    SLACK_WEBHOOK_URI = os.environ["SLACK_WEBHOOK_URI"]