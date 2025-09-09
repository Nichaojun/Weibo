import os
import numpy as np
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# def box_count(data, eps):
#     m, n = data.shape
#     bins = [np.arange(0, 1, eps) for i in range(n)]
#     box_indices = np.empty((m, n), dtype=int)
#     for i in range(n):
#         b = np.copy(bins[i])
#         box_indices[:, i] = np.digitize(data[:, i], b)
#     unique_boxes = set(map(tuple, box_indices))
#     return len(unique_boxes)


# def box_count(data, eps):
#     """
#     data: shape (m, n)，m个点、n维
#     eps : 盒子的边长
#     return: 非空盒子的坐标数组（去重后）
#     """
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

def box_count(data, eps):
    data = np.clip(data, 0, 1 - 1e-6)  
    box_indices = np.floor(data / eps).astype(int)
    box_indices = np.round(box_indices, decimals=6)  
    unique_boxes = np.unique(box_indices, axis=0)
    return unique_boxes.shape[0]


def estimate_box_dimension(data, epsilons, year, output_folder="output/wb"):
    output_folder = output_folder+f'/{year}'
    os.makedirs(output_folder, exist_ok=True)
    counts = []

    with ThreadPoolExecutor(max_workers=64) as executor:
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
    first_max_index = counts_list.index(max_value)  # 第一个 11 的索引是 9
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
    txt_path = os.path.join(output_folder, f"out.txt")
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
    img_path = os.path.join(output_folder, f"out.jpg")
    plt.savefig(img_path)
    plt.close()

    print(f"\nSaved: {img_path}, {txt_path}")
    print(f"Estimated Box-counting Dimension: {dimension:.4f}")
    return dimension


if __name__ == "__main__":
    # 示例：生成随机数据，100个用户，50个话题（观点值在 0~5 之间）
    np.random.seed(42)

    years = [2021]
    for year in years:
        print(year)
        matrix_data_path = f"/lustre/home/2401213204/new-fenxin/matrix/weibo/{year}.npz"
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

        # 设置 epsilon 值
        epsilons = np.arange(0.9, 1.0, 0.0005)
        epsilons = epsilons[::-1]  # 反转数组

        data_normalized = data_normalized.T
        dimension = estimate_box_dimension(data_normalized, epsilons, year=year)