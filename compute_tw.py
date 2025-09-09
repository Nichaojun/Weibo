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

def estimate_box_dimension(data, epsilons, year, output_folder="output/tw/"):
    output_folder = output_folder + f'/{year}'
    os.makedirs(output_folder, exist_ok=True)
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
    img_path = os.path.join(output_folder, f"{year}.jpg")
    plt.savefig(img_path)
    plt.close()

    print(f"\nSaved: {img_path}, {txt_path}")
    print(f"Estimated Box-counting Dimension: {dimension:.4f}")
    return dimension


if __name__ == "__main__":
    np.random.seed(42)

    years = [2021]
    for year in years:
        print(year)
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
            

        # 设置 epsilon 值
        epsilons = np.arange(0.7, 1.0, 0.0005) #0.01基本无法搜索到什么结果 
        epsilons = epsilons[::-1]  # 反转数组

        data_normalized = data_normalized.T
        dimension = estimate_box_dimension(data_normalized, epsilons, year=2021)