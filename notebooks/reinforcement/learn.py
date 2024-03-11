from reinforcement.env import StockTradingEnv
from reinforcement.api import stock_pool_prices
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment, batched_py_environment
from tf_agents.networks import network
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.policies import random_tf_policy


def validate_environments(environments, num_episodes=5):
    for env in environments:
        env = tf_py_environment.TFPyEnvironment(env)
        time_step_spec = env.time_step_spec()
        action_spec = env.action_spec()
        random_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)
        time_step = env.reset()
        for _ in range(num_episodes):
            action_step = random_policy.action(time_step)
            time_step = env.step(action_step.action)
            if time_step.is_last():
                time_step = env.reset()


def get_env(train_stock_pool_id, val_stock_pool_id, ic_id, model_name=''):
    print(f"请求 train pool: {train_stock_pool_id}")
    stock_data_for_training = stock_pool_prices(train_stock_pool_id, ic_id)
    print(f"请求 val pool: {val_stock_pool_id}")
    stock_data_for_eval = stock_pool_prices(val_stock_pool_id, ic_id)
    print("数据请求完毕，准备对应股票的环境列表")
    # 准备对应股票的环境列表
    train_py_envs = [StockTradingEnv(data, model_name=f'{model_name}_trading') for data in stock_data_for_training]
    eval_py_envs = [StockTradingEnv(data, model_name=f'{model_name}_val') for data in stock_data_for_eval]
    print("股票环境列表创建完毕，创建批量训练和评估环境")
    # 创建批量训练和评估环境
    train_env = tf_py_environment.TFPyEnvironment(batched_py_environment.BatchedPyEnvironment(train_py_envs))
    eval_env = tf_py_environment.TFPyEnvironment(batched_py_environment.BatchedPyEnvironment(eval_py_envs))
    print("确保所有环境是有效的")
    # validate_environments(train_py_envs)
    # validate_environments(eval_py_envs)
    print("环境创建完毕")
    return train_env, eval_env, train_py_envs, eval_py_envs


class LstmQNetwork(network.Network):

    def __init__(self, input_tensor_spec, action_spec, lstm_units=(100, 100), fc_layer_params=(100, 100),
                 dropout_rate=0.5):
        super(LstmQNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
        )

        self.layer_norm = tf.keras.layers.LayerNormalization()

        # LSTM 层
        self.lstm_layers = [tf.keras.layers.LSTM(num_units, return_sequences=True if i < len(lstm_units) - 1 else False,
                                                 return_state=True) for i, num_units in enumerate(lstm_units)]

        # 全连接层
        self.dense_layers = [
            tf.keras.layers.Dense(num_units, activation='elu', kernel_regularizer=tf.keras.regularizers.L2(0.001))
            for num_units in fc_layer_params
        ]

        # Dropout层为正则化技术
        self.dropout_layers = [
            tf.keras.layers.Dropout(rate=dropout_rate)
            for _ in fc_layer_params
        ]

        # 输出层
        self.output_layer = tf.keras.layers.Dense(units=(action_spec.maximum - action_spec.minimum + 1))

    def call(self, inputs, step_type, network_state=(), training=False):
        # 可选：对输入进行归一化
        # norm_inputs = self.layer_norm(inputs)
        norm_inputs = inputs

        # LSTM 层
        x = norm_inputs
        for lstm_layer in self.lstm_layers:
            x, _state_h, _state_c = lstm_layer(x)

        # 应用 Dropout 正则化
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            x = dense_layer(x)
            x = dropout_layer(x, training=training)

        # 输出层
        output = self.output_layer(x)

        return output, network_state

    def copy(self, **kwargs):
        kwargs.pop('name', None)
        return super(LstmQNetwork, self).copy(**kwargs)


# 定义数据收集函数
def collect_step(environment, policy, replay_buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)


def get_net(train_env):
    # 设置LSTM网络
    tf_env = train_env
    input_tensor_spec = tensor_spec.from_spec(tf_env.observation_spec())
    time_step_spec = ts.time_step_spec(input_tensor_spec)
    action_spec = tensor_spec.from_spec(tf_env.action_spec())

    # 构建自定义的LSTM网络
    lstm_q_net = LstmQNetwork(
        input_tensor_spec,
        action_spec,
        lstm_units=(100,),
        fc_layer_params=(100,)
    )

    initial_learning_rate = 1e-3
    decay_steps = 10000
    decay_rate = 0.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # 设置优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 设置训练步数计数器
    train_step_counter = tf.Variable(0)

    # 初始化DQN代理 (你可能需要修改参数，以适应你的具体环境)
    agent = dqn_agent.DqnAgent(
        time_step_spec,
        action_spec,
        q_network=lstm_q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )

    agent.initialize()

    # 初始化Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=100000
    )

    # 初始化数据收集策略，这里使用代理的收集策略
    collect_policy = agent.collect_policy

    # 收集初始数据
    init_collect_steps = 100  # 根据需求调整
    for _ in range(init_collect_steps):
        collect_step(train_env, collect_policy, replay_buffer)

    # 准备数据集和迭代器，用于训练
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    return agent, iterator, replay_buffer
