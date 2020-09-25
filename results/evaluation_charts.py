import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

def make_results():
    all_results = get_results_from_pickle_files()
    for task, results in all_results:
        x_values, y_values = extract_plot_data(results)
        macd = get_macd(y_values)
        max_value = max(y_values)
        max_id = [i for i, j in enumerate(y_values) if j == max(y_values)][0]
        exact, = plt.plot(x_values, y_values, label="Exact results")
        moving_avg, = plt.plot(x_values, macd, label="moving average")
        max_value_dot, = plt.plot(max_id, max_value, 'ro')
        plt.ylabel("results")
        plt.xlabel("training run nr.")
        ticks = int(len(x_values) / 10)
        plt.xticks(np.arange(min(x_values), max(x_values) + 1, ticks))
        plt.title(task)
        plt.legend([exact, moving_avg, max_value_dot], ['Exact results', 'Moving Average', f"Max Value: ({max_id}, {round(max_value, 4)})"])
        plt.savefig(f"charts/{task}.png")
        plt.close()

def get_results_from_pickle_files() -> list:
    all_results = []
    files = os.listdir("files")
    for pkl_file in files:
        path_to_pkl_file = f"files/{pkl_file}"
        with open(path_to_pkl_file, 'rb') as f:
            task = extract_task_name(pkl_file)
            all_results.append((task, pickle.load(f)))
    return all_results

def extract_task_name(file_name: str) -> str:
    pattern = "results-(.*?).pkl"
    key = re.search(pattern, file_name).group(1)
    return key

def extract_plot_data(results: dict):
    x_values = [int(re.search(r"\d+", key).group(0)) for key in list(results.keys())]
    y_values = [result_dict.get("result") for result_dict in list(results.values())]
    return x_values, y_values

def get_macd(results: list, window_size: int = 25):
    left_pointer = 0
    if len(results) > window_size:
        pass
    else:
        window_size = int(len(results)/2)
    right_pointer = window_size
    macd = list(np.zeros(window_size - 1))
    for single_macd in range(window_size, len(results)+1):
        values = results[left_pointer:right_pointer]
        macd.append(average(values))
        left_pointer += 1
        right_pointer += 1
    return macd

def average(values: list):
    return sum(values) / len(values)

if __name__ == "__main__":
    make_results()