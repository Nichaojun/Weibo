import os
import numpy as np
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def box_count(data, eps):
    data = np.clip(data, 0, 1 - 1e-6)  
    box_indices = np.floor(data / eps).astype(int)
    box_indices = np.round(box_indices, decimals=6)  
    unique_boxes = np.unique(box_indices, axis=0)
    return unique_boxes.shape[0]

# def box_count(data, eps):
#     # 每一维的最小值和最大值
#     mins = np.min(data, axis=0)
#     maxs = np.max(data, axis=0)
#     span = maxs - mins
#     # 平移到各维min为零点
#     shifted = data - mins
#     # 每一维需要的盒子数（ceil确保覆盖到最大值）
#     n_bins = np.ceil(span / eps).astype(int)
#     n_bins = np.where(n_bins == 0, 1, n_bins)  # 常数维至少一个格子
#     # 每个点的盒子编号
#     idx = np.floor(shifted / eps).astype(int)
#     # clip到范围内
#     idx = np.clip(idx, 0, n_bins - 1)
#     # 返回所有非空盒子的坐标
#     return np.unique(idx, axis=0).shape[0]

def find_epsilon_for_target_box_count(data, target_box_count, epsilon_min=0.01, epsilon_max=1.0, tolerance=1, mid_cache={}):
    left = epsilon_min
    right = epsilon_max
    best_eps = left  # 初始化最大满足条件的epsilon为最小值
    box_count_current = 0
    times = 0
    
    while right - left > 1e-6 or box_count_current != target_box_count:  # 搜索直到 epsilon 范围足够小
        if times > 50:
            break
        mid = (left + right) / 2
        times += 1
        # 检查mid是否已经计算过，如果计算过直接使用缓存的结果
        if mid in mid_cache:
            box_count_current = mid_cache[mid]
        else:
            box_count_current = box_count(data, mid)  # 获取当前 epsilon 下的盒子数
            mid_cache[mid] = box_count_current  # 缓存计算结果
        
        if box_count_current >= target_box_count:  # 盒子数满足条件
            best_eps = mid  # 记录当前 epsilon
            left = mid  # 尝试增大 epsilon，继续收缩左边界
        else:  # 盒子数不满足条件
            right = mid  # 减小 epsilon，收缩右边界
        print(target_box_count,left,right,flush=True)
    if times > 50:
        return None
    
    return best_eps

if __name__ == "__main__":
    matrix_data_path = f"/lustre/home/2401213204/new-fenxin/matrix/twitter_matrix/2021_final.npz"
    data = np.load(matrix_data_path)
    
    data = data['matrix']
    data = np.nan_to_num(data, nan=0)
    data = np.clip(data, -2, 2)

    data_min = data.min()
    data_max = data.max()
    if data_max - data_min != 0: 
        data_normalized = (data - data_min) / (data_max - data_min)
    else:
        data_normalized = np.full(data.shape, 0.5)
    data_normalized = data_normalized.T
    max_box_count = data_normalized.shape[0]

    mid_cache = {}
    result = {}
    # Loop through target box counts from max_box_count down to 1
    for target_box_count in range(max_box_count, 0, -1):
        best_eps = find_epsilon_for_target_box_count(data_normalized, target_box_count,mid_cache=mid_cache)
        if best_eps != None:
            result[target_box_count] = best_eps
    result[1] = 1
    print(result)

    counts_list = list(result.keys())
    epsilons_list = list(result.values())
    log_eps = np.log(1 / np.array(epsilons_list))
    log_counts = np.log(counts_list)
    model = LinearRegression(fit_intercept=False)
    model.fit(log_eps.reshape(-1, 1), log_counts.reshape(-1, 1))
    dimension = model.coef_[0].item()
    print(f"Estimated Box-counting Dimension: {dimension:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(log_eps, log_counts, 'o', label='Data')
    x_fit = np.array([min(log_eps), max(log_eps)])
    y_fit = model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, '-', color='red', label=f'Fit (slope = {dimension:.2f})')
    plt.xlabel("log(1 / ε)")
    plt.ylabel("log(N(ε))")
    plt.title("Box-counting Dimension Estimation")
    plt.legend()
    plt.grid(True)
    output_folder='output'
    tpt = f'{output_folder}/tw/2021_bisection/'
    os.makedirs(tpt, exist_ok=True)
    plt.savefig(tpt+'/out.jpg')
