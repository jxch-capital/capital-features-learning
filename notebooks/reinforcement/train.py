from tf_agents.metrics import tf_metrics

# from reinforcement.learn import iterator, collect_step, agent, eval_env, train_env, replay_buffer
from reinforcement.learn import collect_step, get_net, get_env
import tensorflow as tf
from tqdm import tqdm
import os
from tf_agents.utils import common
from tf_agents.policies import policy_saver


# 函数收集一定数量的步骤并添加到Replay Buffer
def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


# 函数评估智能体的性能
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in tqdm(range(num_episodes)):
        time_step = environment.reset()
        episode_return = 0.0

        while not tf.reduce_all(time_step.is_last()):  # Use tf.reduce_all()
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += tf.reduce_sum(time_step.reward).numpy()  # Use tf.reduce_sum()
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return


def get_checkpointer(model_name, agent,
                     checkpoint_dir=r'/data/checkpoint/',
                     policy_dir=r'/data/policies/'):
    # 指定保存模型和策略的路径
    checkpoint_dir = checkpoint_dir + model_name + r'/'
    policy_dir = policy_dir + model_name + r'/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)

    # 创建Checkpoint对象
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=10,
        agent=agent,
        policy=agent.policy,
        global_step=agent.train_step_counter
    )

    # 创建PolicySaver对象
    train_policy_saver = policy_saver.PolicySaver(agent.policy)

    # 进行保存
    def save_train_checkpointer():
        train_checkpointer.save(agent.train_step_counter)

    def save_train_policy_saver():
        train_policy_saver.save(policy_dir)

    return save_train_checkpointer, save_train_policy_saver


def train(train_env, eval_env, train_py_envs, eval_py_envs, model_name, epoch=100, num_iterations=6000,
          collect_steps_per_iteration=1,
          log_interval=100, eval_interval=100, early_stopping_patience=20):
    # 主训练循环
    # num_iterations 训练总步骤数，根据需要进行调整
    # collect_steps_per_iteration 每次迭代后都收集一定的步骤到Replay Buffer
    # log_interval 每隔1000步打印日志
    # eval_interval 每隔1000步评估模型的性能

    print("构建网络")
    agent, iterator, replay_buffer = get_net(train_env)
    save_train_checkpointer, save_train_policy_saver = get_checkpointer(model_name, agent)

    print("初始评估")
    # 初始评估
    # avg_return = compute_avg_return(eval_env, agent.policy, 10)

    average_rp_list = []
    average_rp_max_list = []
    average_rp_min_list = []
    returns = {
        "average_rp": average_rp_list,
        "average_max_rp": average_rp_max_list,
        "average_min_rp": average_rp_min_list,
    }

    max_average_rp = 0
    max_average_rp_max = 0
    max_average_rp_min = 0
    print("开始训练")
    for ie in range(epoch):
        total_loss = 0
        with tqdm(total=len(range(num_iterations))) as progress_bar:
            for index in range(num_iterations):
                # 从训练环境中收集一些步骤（使用代理的collect_policy）并存储到replay buffer。
                collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

                # 每次收集后都从replay buffer中取出一小批数据用于学习。
                experience, _ = next(iterator)
                train_loss = agent.train(experience).loss
                total_loss = total_loss + float(train_loss)

                # 训练步数
                step = agent.train_step_counter.numpy()

                # 环境性能
                rp_list = [env.return_performance() for env in train_py_envs]
                average_rp = sum(rp_list) / len(rp_list) if rp_list else 0

                metrics = {
                    'epoch': ie + 1,
                    'step': step,
                    'loss': float(total_loss) / float(index + 1),
                    'average_rp': average_rp,
                    'average_rp_max': max(rp_list),
                    'average_rp_min': min(rp_list),
                }

                progress_bar.set_postfix(metrics)
                progress_bar.update(1)

                # 每隔一定步数评估模型性能，并记录返回。
                # if step % eval_interval == 0:
                # avg_return = compute_avg_return(eval_env, agent.policy, 10)
                # print('step = {0}: Average Return = {1}'.format(step, avg_return))
                # returns.append(avg_return)

            rp_list = [env.last_return_performance() for env in train_py_envs]
            average_rp = sum(rp_list) / len(rp_list) if rp_list else 0
            average_rp_list.append(average_rp)
            max_arp = max(rp_list)
            min_arp = min(rp_list)
            average_rp_max_list.append(max_arp)
            average_rp_min_list.append(min_arp)
            print(
                f"Epoch {ie + 1} finished, average_rp: {average_rp}%, average_rp_max: {max_arp}%, average_rp_min: {min_arp}%")

            is_save = False
            if len(average_rp_list) == 1:
                print(f"average_rp: {average_rp}%, 保存模型参数")
                max_average_rp = average_rp
                max_average_rp_max = max_arp
                max_average_rp_min = min_arp
                is_save = True
            if average_rp > max_average_rp:
                print(f"average_rp 从 {max_average_rp}% 提高到 {average_rp}%")
                max_average_rp = average_rp
                is_save = True
            if max_arp > max_average_rp_max:
                print(f"average_rp_max 从 {max_average_rp_max}% 提高到 {max_arp}%")
                max_average_rp_max = max_arp
                is_save = True
            if min_arp > max_average_rp_min:
                print(f"average_rp_min 从 {max_average_rp_min}% 提高到 {min_arp}%")
                max_average_rp_min = min_arp
                is_save = True

            if is_save:
                print("保存模型")
                save_train_checkpointer()
                save_train_policy_saver()
            elif len(average_rp_list) > early_stopping_patience and len(
                    average_rp_max_list) > early_stopping_patience and len(
                average_rp_min_list) > early_stopping_patience and max(
                average_rp_list[-early_stopping_patience:]) < max_average_rp and max(
                average_rp_max_list[-early_stopping_patience:]) < max_average_rp_max and max(
                average_rp_min_list[-early_stopping_patience:]) < max_average_rp_min:
                print(
                    f"连续{early_stopping_patience}轮没有改进，早停：max_average_rp: {max_average_rp}%, average_rp_max: {max_arp}%, average_rp_min: {min_arp}%")
                break

    return returns


def env_train(train_pool_id, val_pool_id, ic_id, model_name, epoch=100, num_iterations=6000):
    train_env, eval_env, train_py_envs, eval_py_envs = get_env(2059184, 2059184, 1329741, model_name=model_name)
    returns = train(train_env, eval_env, train_py_envs, eval_py_envs, model_name=model_name, epoch=epoch,
                    num_iterations=num_iterations)
    return returns
