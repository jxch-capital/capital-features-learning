import requests
import json

prediction_data_url = "http://jiangxicheng.site:18888/learning_data/prediction_data"
train_data_url = "http://jiangxicheng.site:18888/learning_data/train_data"

payload = json.dumps({
  "trainConfigId": 35602,
  "code": "SPY",
  "start": "2020-01-01",
  "end": "2022-01-01"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", train_data_url, headers=headers, data=payload)

print(response.text)
