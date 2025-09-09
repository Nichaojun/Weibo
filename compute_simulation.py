import os
import numpy as np
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

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

def estimate_box_dimension(data, epsilons, output_folder="None",year='None'):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f'{output_folder}/{year}', exist_ok=True)
    counts = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for eps in epsilons:
            futures.append(executor.submit(box_count, data, eps))
        
        for i, future in enumerate(futures):
            eps = epsilons[i]
            count = future.result()
            print(f"epsilon = {eps:.4f}, boxes = {count}")
            counts.append((eps, count))  # 保存为元组

    # 拆分保存为两个列表
    epsilons_list = [x[0] for x in counts]
    counts_list = [x[1] for x in counts]

    max_value, _ = data.shape
    first_max_index = counts_list.index(max_value) 
    counts_list = counts_list[:first_max_index+1]
    epsilons_list = epsilons_list[:first_max_index+1]


    # 转换为 log 值
    log_eps = np.log(1 / np.array(epsilons_list))
    log_counts = np.log(counts_list)

    # 线性拟合
    model = LinearRegression(fit_intercept=False)
    model.fit(log_eps.reshape(-1, 1), log_counts.reshape(-1, 1))
    dimension = model.coef_[0].item()

    # 保存结果到 txt 文件
    txt_path = os.path.join(f'{output_folder}/{year}', f"out.txt")
    with open(txt_path, "w") as f:
        f.write(f"Year: {year}\n")
        f.write(f"Estimated Box-counting Dimension: {dimension:.4f}\n")
        f.write("epsilon,box_count\n")
        for eps, count in zip(epsilons_list, counts_list):
            f.write(f"{eps:.4f},{count}\n")

    # 可视化 log-log 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(log_eps, log_counts, 'o', label='Data')
    x_fit = np.array([min(log_eps), max(log_eps)])
    y_fit = model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, '-', color='red', label=f'Fit (slope = {dimension:.2f})')
    plt.xlabel("log(1 / ε)")
    plt.ylabel("log(N(ε))")
    plt.title(f"Box-counting Dimension Estimation ({year})")
    plt.legend()
    plt.grid(True)

    # 保存图像
    img_path = os.path.join(f'{output_folder}/{year}', f"out.jpg")
    plt.savefig(img_path)
    plt.close()

    print(f"\nSaved: {img_path}, {txt_path}")
    print(f"Estimated Box-counting Dimension: {dimension:.4f}")
    return dimension



if __name__ == "__main__":
    # 示例：生成随机数据，10000个用户，10个话题（观点值在 0~1 之间）
    np.random.seed(42)

    m, n = 10, 100000
    T = 1.0
    M = 0.7
    J = 10

    size = m * n

    # 70% 固定 0.5 这个模拟大多数用户不发言
    num_fixed = int(size * M)
    fixed_part = np.full(num_fixed, 0.5) 

    # 30% 随机 [0,1)
    num_random = size - num_fixed
    random_part = np.random.rand(num_random) * T

    # 拼接后打乱
    all_values = np.concatenate([fixed_part, random_part])
    np.random.shuffle(all_values)

    # reshape 成矩阵
    matrix = all_values.reshape(m, n)

    #假设有数量为J的用户有极端情况，在一个话题下面发言为支持 在另外一个话题下发言为反对
    for p in range(0, J):
        user_1 = np.random.randint(0, m)
        matrix[user_1][0] = 1
        user_2 = np.random.randint(0, m)
        while user_2 == user_1:  
            user_2 = np.random.randint(0, m)
        
        matrix[user_2][0] = 0

    epsilons = np.arange(0.99, 1.0, 0.0001) 
    epsilons = np.append(epsilons,1.0)
    epsilons = epsilons[::-1]
    dimension = estimate_box_dimension(matrix, epsilons,output_folder='output',year='simulation')