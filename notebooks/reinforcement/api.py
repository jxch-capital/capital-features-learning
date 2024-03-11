import requests

base_url = "/stock_pool_/stock_pool_prices_by_ic?stockPoolId="

dev_host = "http://127.0.0.1:18088"
dev_docker_host = "http://host.docker.internal:18088"
prod_host = "http://amd2.jiangxicheng.site:18888"

dev_host_url = dev_host + base_url
dev_docker_host_url = dev_docker_host + base_url
prod_host_url = prod_host + base_url

payload = {}
headers = {}


def stock_pool_prices(stock_pool_id, ic_id, max_length=200, url=dev_docker_host_url):
    response = requests.request("POST",
                                url + str(stock_pool_id) + "&icId=" + str(ic_id) + "&maxLength=" + str(max_length),
                                headers=headers, data=payload)
    return response.json()
