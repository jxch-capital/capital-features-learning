import requests
import json
from datetime import datetime

dev_host = "http://127.0.0.1:18088"
dev_docker_host = "http://host.docker.internal:18088"
prod_host = "http://amd2.jiangxicheng.site:18888"

train_data_url_path = "/learning_data/train_data"
prediction_data_url_path = "/learning_data/prediction_data"
kline_history_url_path = "/stock_/history"

dev_train_data_url = dev_host + train_data_url_path
prod_train_data_url = prod_host + train_data_url_path
dev_prediction_data_url = dev_host + prediction_data_url_path
prod_prediction_data_url = prod_host + prediction_data_url_path
docker_train_data_url = dev_docker_host + train_data_url_path
docker_prediction_data_url = dev_docker_host + prediction_data_url_path
dev_kline_history_url = dev_host + kline_history_url_path
docker_kline_history_url = dev_docker_host + kline_history_url_path
prod_kline_history_url = prod_host + kline_history_url_path

headers = {
    'Content-Type': 'application/json'
}


def get_prediction_data(prediction_api=prod_prediction_data_url, train_config_id=35602, code="SPY", start="2020-01-01",
                        end=datetime.now().strftime("%Y-%m-%d")):
    payload = json.dumps({
        "trainConfigId": train_config_id,
        "code": code,
        "start": start,
        "end": end
    })

    response = requests.request("POST", prediction_api, headers=headers, data=payload)
    return json.loads(response.text)


def get_train_data(train_api=prod_train_data_url, train_config_id=35602):
    payload = json.dumps({
        "trainConfigId": train_config_id
    })

    response = requests.request("POST", train_api, headers=headers, data=payload)
    json_res = json.loads(response.text)
    return json_res['knodeTrains']


def get_kline_histroy(url=prod_kline_history_url, code="SPY", start="2023-07-25",
                      end=datetime.now().strftime("%Y-%m-%d"), interval="1d", engine="YAHOO_CHART"):
    payload = json.dumps({
        "code": code,
        "start": start,
        "end": end,
        "interval": interval,
        "engine": engine
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)
