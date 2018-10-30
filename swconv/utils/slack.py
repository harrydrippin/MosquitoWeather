"""File for notifying training informations on Slack."""
from config import Config
import requests
import json

def notify(content):
    """
    Slack 채널에 내용을 게시합니다.
    :param content 게시할 내용
    """
    requests.post(Config.SLACK_WEBHOOK_URI, 
        data=json.dumps(content), 
        headers={
        "Content-Type": "application/json"
        }
    );

def notify_complete(_format, _scaler, _pca, _model, _level, mse, additional, time, memo=None):
    """
    Slack 채널에 완료 사실을 알립니다.
    """
    if memo is None: memo = "수행"
    
    if _level:
        additional *= 100
    
    content = {
        "attachments": [
            {
                "fallback": f"학습 {memo} 완료 ({time:2.2f}초 소요)",
                "color": "#36a64f",
                "author_name": "Seunghwan Hong",
                "title": f"학습 {memo} 완료 ({time:2.2f}초 소요)",
                "text": f"*Shape*: {_format}안\n*Scale*: {_scaler}\n*PCA*: {_pca}\n*Model*: {_model}\n*Leveling*: {_level}",
                "mrkdwn_in": ["text"],
                "fields": [
                    {
                        "title": "MSE",
                        "value": f"{mse:6.6f}",
                        "short": True
                    }
                ],
                "footer": "SWConv ML Engine",
                "footer_icon": "https://avatars.slack-edge.com/2018-10-28/466336881570_689e676a4c54c5283567_72.png",
                "ts": 123456789
            }
        ]
    }

    content["attachments"][0]["fields"].append({
        "title": "Accuracy" if _level is True else "R2 Score",
        "value": f"{additional:3.3f}" + ("%" if _level is True else ""),
        "short": True
    })

    notify(content)