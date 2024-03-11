import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import datetime
import os
import multiprocessing
import matplotlib.colors as mcolors

semaphore = multiprocessing.BoundedSemaphore(3)


def normalized(arr):
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 避免除零错误
    return (arr - min_vals) / range_vals


class StockTradingEnv(py_environment.PyEnvironment):
    _num = 1

    def __init__(self, stock_data, initial_cash=100000, history_length=50, reward_lookahead=6,
                 filename=datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'), plt_dir=r'/data/history/', model_name=''):
        self._stock_data = np.array(stock_data)
        self._history_length = history_length
        self._current_step = history_length
        self._initial_cash = initial_cash
        self._cash_in_hand = initial_cash
        self._num_stocks_owned = 0
        self._start_step = history_length
        self._reward_lookahead = reward_lookahead
        self._total_asset_record = [initial_cash] * history_length  # 记录每一步的总资产以计算奖励
        self._filename = str(StockTradingEnv._num) + "_" + filename
        StockTradingEnv._num += 1
        self._episode_counter = 0  # 跟踪轮数的计数器
        self._trades = []  # 交易详情
        self._plt_dir = plt_dir + model_name + '/'
        os.makedirs(self._plt_dir, exist_ok=True)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=10, name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=(history_length, 7 + 4 + 5), dtype=np.float32, name='observation')  # 修改以包含额外的特征

        self._episode_ended = False
        self._epoch_return_on_assets_performance = []
        self._return_performance = [0] * history_length

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _calculate_reward(self, action, transacted_shares):
        # Use future data to calculate reward.
        future_steps = min(len(self._stock_data) - self._current_step, self._reward_lookahead)
        if future_steps < self._reward_lookahead:
            self._episode_ended = True
            return 0

        future_data = self._stock_data[self._current_step:self._current_step + future_steps]
        current = self._stock_data[self._current_step][3]
        future_max = future_data[:, 1].max()
        future_min = future_data[:, 2].min()

        if action == 0:  # 观望
            # 情景1: 有头寸观望时，资产缩水的惩罚
            if self._num_stocks_owned > 0 and future_max < current and (current - future_min) / current > 0.15:
                return -(current - future_min) / current * self._num_stocks_owned

            # 情景2: 有现金可购股但未购，错过资产增长的惩罚
            current_buying_power = self._cash_in_hand / current
            if current_buying_power >= 25 * 10 and future_min > current and (future_max - current) / current > 0.15:
                return -(future_max - current) / current * current_buying_power

            ratio = abs((future_max - current) / (current - future_min))
            return min(ratio, 1 / ratio) * max(current_buying_power, self._num_stocks_owned)
        if action <= 5:  # 做多/买入
            if transacted_shares == 0:
                return -1 * action
            max_favorable_variation = future_max - current
            max_adverse_variation = current - future_min
            return ((max_favorable_variation - max_adverse_variation) / abs(max_adverse_variation) if abs(
                max_adverse_variation) != 0 else 1) * action
        elif action <= 10:  # 做空/卖出
            if transacted_shares == 0:
                return -1 * (action - 5)
            max_favorable_variation = current - future_min
            max_adverse_variation = future_max - current
            return ((max_favorable_variation - max_adverse_variation) / abs(max_adverse_variation) if abs(
                max_adverse_variation) != 0 else 1) * (action - 5)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        transacted_shares = 0  # 这里存储实际交易的股票数量
        current_stock_price = self._stock_data[self._current_step][3]

        if action == 0:  # 观望
            pass
        elif action <= 5:  # 买入
            # 根据动作买入相应的仓位
            order_fraction = 1 / (action * action)  # 分别对应1, 1/4, 1/9, 1/16, 1/25
            # 计算可以购买的最大股数
            max_shares_to_buy = int(self._cash_in_hand / current_stock_price)
            shares_to_buy = int(max_shares_to_buy * order_fraction)
            if shares_to_buy > 0:
                self._num_stocks_owned += shares_to_buy
                self._cash_in_hand -= shares_to_buy * current_stock_price
                self._trades.append((self._current_step, 'buy', shares_to_buy, order_fraction))
                transacted_shares = shares_to_buy  # 保存买入的股票数量
        elif action <= 10:  # 卖出
            # 根据动作卖出相应的仓位
            order_fraction = 1 / ((action - 5) * (action - 5))  # 分别对应1, 1/4, 1/9, 1/16, 1/25
            # 计算可以卖出的最大股数
            max_shares_to_sell = self._num_stocks_owned
            shares_to_sell = int(max_shares_to_sell * order_fraction)
            if shares_to_sell > 0:
                self._num_stocks_owned -= shares_to_sell
                self._cash_in_hand += shares_to_sell * current_stock_price
                self._trades.append((self._current_step, 'sell', shares_to_sell, order_fraction))
                transacted_shares = shares_to_sell  # 保存卖出的股票数量

        # 更新状态和步数
        self._current_step += 1
        if self._current_step >= len(self._stock_data) - self._reward_lookahead:
            self._episode_ended = True

        # 记录图表数据
        new_total_asset = self._cash_in_hand + self._num_stocks_owned * current_stock_price
        reward = self._calculate_reward(action, transacted_shares)
        self._total_asset_record.append(new_total_asset)
        self._return_performance.append(self.return_performance())

        if self._episode_ended:
            self._episode_counter += 1  # 跟踪轮数的计数器增加
            self._draw_charts()  # 绘制并保存K线图和资金变化曲线
            epoch_reward = self.return_performance()
            self._epoch_return_on_assets_performance.append(epoch_reward)
            return ts.termination(np.array(self._state, dtype=np.float32), epoch_reward)
        else:
            self._state = self._get_observation()
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)

    def return_performance(self):
        start_price = self._stock_data[self._start_step][3]
        current_stock_price = self._stock_data[self._current_step][3]
        new_total_asset = self._cash_in_hand + self._num_stocks_owned * current_stock_price
        if current_stock_price <= start_price:
            return 100 * (((new_total_asset - self._initial_cash) / self._initial_cash) - (
                    (current_stock_price - start_price) / start_price))
        else:
            return 100 * (((new_total_asset - self._initial_cash) / self._initial_cash) - (
                    (current_stock_price - start_price) / start_price))

    def last_return_performance(self):
        return self._epoch_return_on_assets_performance[-1]

    def _reset(self):
        self._current_step = self._start_step
        self._cash_in_hand = self._initial_cash
        self._num_stocks_owned = 0
        self._episode_ended = False
        self._total_asset_record = [self._initial_cash] * self._history_length  # 重置总资产记录
        self._return_performance = [0] * self._history_length
        self._trades = []  # 重置贸易详情
        self._state = self._get_observation()
        self._sum_action = 1
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _get_observation(self):
        # 获取历史股价数据
        obs_klines = self._stock_data[(self._current_step - self._history_length):self._current_step, :4]
        obs_indicators = self._stock_data[(self._current_step - self._history_length):self._current_step, 4:]

        full_history_klines = self._stock_data[(self._start_step - self._history_length):self._current_step, :4]
        compressed_klines = self._compress_klines(full_history_klines, self._history_length)

        # 合并K线数据与指标数据
        obs = np.hstack((normalized(obs_klines), normalized(compressed_klines), normalized(obs_indicators)))

        # 计算新的两个特征
        current_stock_price = self._stock_data[self._current_step - 1][3]  # 获取最新的股价
        max_buyable_shares = min(25, self._cash_in_hand / current_stock_price)  # 可购买的最大股数，最大25股
        buy_power_feature = min(0.25, max_buyable_shares * 0.01)  # 可购买的股数转换为特征值

        max_owns = min(25, self._num_stocks_owned)  # 持有的最大股数，最大25股
        own_stock_feature = min(0.25, max_owns * 0.01)  # 持有的股数转换为特征值

        # 将新的特征填充到历史长度的列
        buy_power_column = np.full((self._history_length, 1), buy_power_feature, dtype=np.float32)
        own_stock_column = np.full((self._history_length, 1), own_stock_feature, dtype=np.float32)

        return_performance_history = np.array(self._return_performance[-self._history_length:],
                                              dtype=np.float32).reshape(-1, 1)
        combined_obs = np.hstack(
            (obs, return_performance_history, buy_power_column, own_stock_column))  # 合并所有列

        return combined_obs

    def _compress_klines(self, klines, compressed_length):
        # This function combines the klines according to the rules for kline aggregation
        num_klines = klines.shape[0]
        interval = num_klines / compressed_length
        compressed_klines = []

        # Starting index for iteration
        start_idx = 0

        for i in range(compressed_length):
            # Calculate the index range for the current bucket of klines to be compressed
            end_idx = int(round((i + 1) * interval))

            # Extract the klines for the current bucket
            bucket_klines = klines[start_idx:end_idx]

            # Aggregate the klines by the rules provided
            open_price = bucket_klines[0, 0]  # First kline’s open
            close_price = bucket_klines[-1, 3]  # Last kline’s close
            high_price = bucket_klines[:, 1].max()  # Highest high in the bucket
            low_price = bucket_klines[:, 2].min()  # Lowest low in the bucket

            # Append this aggregated kline to the list
            compressed_klines.append([open_price, high_price, low_price, close_price])

            # Move the start index for the next iteration
            start_idx = end_idx

        return np.array(compressed_klines)

    def _draw_charts(self):
        # 为绘制K线图准备数据
        lengths = self._current_step - self._history_length
        stock_df = pd.DataFrame(self._stock_data[self._history_length:self._current_step, :4],
                                columns=['Open', 'High', 'Low', 'Close'])
        stock_df['Date'] = pd.date_range(start='1/1/2000', periods=lengths, freq='B')
        stock_df = stock_df.set_index('Date')

        # 初始化买入和卖出列表以及颜色
        buys = [np.nan] * lengths
        sells = [np.nan] * lengths
        buy_colors = [(0, 0, 0, 0)] * lengths
        sell_colors = [(0, 0, 0, 0)] * lengths

        # 颜色映射函数，根据order_fraction映射颜色深度
        def get_color_depth(order_fraction_):
            max_color_value = 0.7  # 最深颜色对应的值
            return max_color_value * order_fraction_

        for trade_step, trade_action, shares, order_fraction in self._trades:
            index = trade_step - self._history_length
            if index < lengths:
                alpha = get_color_depth(order_fraction)
                if trade_action == 'buy':
                    buys[index] = stock_df['Low'].iloc[index] * 0.98
                    buy_colors[index] = mcolors.to_rgba('blue', alpha=alpha)
                elif trade_action == 'sell':
                    sells[index] = stock_df['High'].iloc[index] * 1.02
                    sell_colors[index] = mcolors.to_rgba('orange', alpha=alpha)

        # 创建apds，并且确保传递的颜色参数不包含nan值
        apds = [mpf.make_addplot(buys, type='scatter', markersize=100, marker='^', color=buy_colors),
                mpf.make_addplot(sells, type='scatter', markersize=100, marker='v', color=sell_colors)]

        def save(_stock_df, _apds, _episode_counter, _plt_dir, _filename, _total_asset_record, _history_length,
                 _return_performance):
            semaphore.acquire()
            mpf.plot(_stock_df, type='candle', addplot=_apds, style='charles',
                     title=f'Stock Trading K - line - Episode {_episode_counter} -资产回报表现 [{_return_performance}%]',
                     figsize=(300, 20),
                     savefig=dict(fname=f"{_plt_dir}{_filename}_{_episode_counter}.png", dpi=100))
            # 绘制和保存资金变化曲线
            plt.figure(figsize=(300, 20))
            plt.plot(_total_asset_record[_history_length:], marker=None, color='blue')
            plt.title(f'Total Asset Value - Episode {_episode_counter} -return_performance [{_return_performance}%]')
            plt.xlabel('Time Step')
            plt.ylabel('Total Asset Value')
            plt.grid(True)
            plt.savefig(f"{_plt_dir}{_filename}_{_episode_counter}_assets.png")
            plt.close('all')
            semaphore.release()

        multiprocessing.Process(target=save,
                                args=(stock_df, apds, self._episode_counter, self._plt_dir, self._filename,
                                      self._total_asset_record, self._history_length,
                                      self.return_performance())).start()
