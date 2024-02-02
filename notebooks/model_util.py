import re
import matplotlib.pyplot as plt


def find_last_epoch_txt_log(log_file_path):
    last_epoch = 0
    epoch_pattern = re.compile(r"Epoch (\d+)")  # 假设每个epoch的日志以"Epoch 数字"开始

    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):  # 反向查找最新的epoch记录
                match = epoch_pattern.search(line)
                if match:
                    last_epoch = int(match.group(1))
                    break
    except FileNotFoundError:
        print("Log file not found. Starting from epoch 0.")

    return last_epoch


def find_attr_txt_log(log_file_path, attr_name):
    attr_values = []
    # 修改正则表达式以匹配 "attr_name: 浮点数" 的模式
    pattern = re.compile(attr_name + r": (\d+\.\d+)")
    with open(log_file_path, 'r') as file:
        for line in file:  # 逐行读取以减少内存占用
            match = pattern.search(line)
            if 'step' in line and match:
                # 将捕获的浮点数添加到结果列表中
                attr_values.append(float(match.group(1)))
    return attr_values


def plot_attr(log_file_path, attrs=None):
    if attrs is None:
        attrs = ['accuracy', 'val_accuracy', 'loss', 'val_loss', ]

    for attr in attrs:
        plt.plot(find_attr_txt_log(log_file_path, attr), label=attr)

    # 添加网格线，但不包括外框
    plt.grid(True, color='gray', linestyle=':', linewidth=0.5)  # 网格线样式

    # 取消外框（实际上是设置外框颜色为背景色使其“消失”）
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)  # 将左侧框线设置为黑色
    plt.gca().spines['bottom'].set_color('black')  # 将底部框线设置为黑色

    plt.legend()
    plt.show()


