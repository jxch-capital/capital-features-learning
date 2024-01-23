import requests
import json
from datetime import datetime

def_host = "http://127.0.0.1:18088"
prod_host = "http://jiangxicheng.site:18888"

train_data_url_path = "/learning_data/train_data"
prediction_data_url_path = "/learning_data/prediction_data"

headers = {
    'Content-Type': 'application/json'
}


def get_prediction_data(train_config_id=35602, code="SPY", start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d")):
    payload = json.dumps({
        "trainConfigId": train_config_id,
        "code": code,
        "start": start,
        "end": end
    })

    response = requests.request("POST", def_host + train_data_url_path, headers=headers, data=payload)
    return json.loads(response.text)


res = get_prediction_data(train_config_id=63068)
print(res)
