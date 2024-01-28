import requests
import json
from datetime import datetime
import pandas as pd

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
    json_res = response.json()
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
    return response.json()


def his_to_df(his):
    # 创建一个DataFrame:
    df = pd.DataFrame(his)

    # 转换’date’列为pandas的Datetime类型:
    df['date'] = pd.to_datetime(df['date'])

    # 设置date列为索引:
    df.set_index('date', inplace=True)

    # 按日期升序排序（如果数据未按日期排序的话）:
    df.sort_index(inplace=True)
    return df


def plot_Y_prediction_and_kline(Y_prediction, df, th = 0.75, color='g'):
    predictions_series = pd.Series(Y_prediction.flatten(), index=df.index)

    signals = np.full(df.shape[0], np.nan)  # 含 NaN 的数组，与 df 的长度一致
    signals[predictions_series > th] = df['high'][predictions_series > th] * 1.01
    markers = mpf.make_addplot(signals, type='scatter', markersize=100, marker='^', color=color)

    mpf.plot(df, type='candle', style='charles', addplot=markers, volume=True, figsize=(24, 16))
