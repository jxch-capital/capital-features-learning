import requests
import json
from datetime import datetime

dev_host = "http://127.0.0.1:18088"
prod_host = "http://amd2.jiangxicheng.site:18888"

train_data_url_path = "/learning_data/train_data"
prediction_data_url_path = "/learning_data/prediction_data"

dev_train_data_url = dev_host + train_data_url_path
prod_train_data_url = prod_host + train_data_url_path
dev_prediction_data_url = dev_host + prediction_data_url_path
prod_prediction_data_url = prod_host + prediction_data_url_path

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
