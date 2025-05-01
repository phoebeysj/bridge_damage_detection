import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import sys
import argparse
import glob
import matplotlib
from scipy.stats import gaussian_kde
import warnings
import matplotlib.colors as colors

# 限制线程数，避免OpenBLAS警告和资源耗尽
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# 设置matplotlib中文字体支持
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    try:
        # 尝试使用系统中文字体
        fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        font_found = False
        
        for font in fonts:
            try:
                matplotlib.rc('font', family=font)
                matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
                plt.rcParams['font.sans-serif'] = [font]
                # 测试中文显示
                plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
                plt.close()
                print(f"成功设置中文字体: {font}")
                font_found = True
                break
            except:
                continue
        
        if not font_found:
            # 如果系统字体不可用，尝试使用matplotlib自带的DejaVu Sans
            matplotlib.rc('font', family='DejaVu Sans')
            matplotlib.rcParams['axes.unicode_minus'] = False
            print("无法找到系统中文字体，使用默认字体")
            print("注: 若需完整中文支持，请安装对应中文字体")
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("继续使用默认字体")

# 在导入后立即设置中文字体
setup_chinese_font()

def visualize_latent_space(file_path, output_dir='./latent_visualizations', 
                           perplexity=30, n_clusters=0, use_pca_first=False,
                           pca_components=50, plot_3d=False, learning_rate='auto',
                           colormap='viridis', specific_key=None, use_env_data=False):
    """读取.pt文件中的潜空间变量并使用t-SNE可视化"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载.pt文件
        data = torch.load(file_path)
        file_name = os.path.basename(file_path).split('.')[0]
        
        # 打印数据类型
        print(f"\n处理文件: {file_name}")
        print(f"数据类型: {type(data)}")
        
        # 如果指定了特定的键，直接使用
        if specific_key and isinstance(data, dict) and specific_key in data:
            if isinstance(data[specific_key], torch.Tensor):
                latent_vars = {specific_key: data[specific_key].detach().cpu().numpy()}
                print(f"使用指定的键: '{specific_key}'")
            else:
                print(f"指定的键 '{specific_key}' 不是张量，无法使用")
                latent_vars = {}
        else:
            # 通过文件名确定潜空间变量
            latent_vars = identify_latent_variables(data, file_name)
        
        if not latent_vars:
            if isinstance(data, dict):
                print("未找到潜空间变量。以下是可用的键：")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: {type(value)} 形状: {value.shape}")
            print("请使用 --key 参数指定要可视化的键名")
            return False
        
        # 如果需要使用环境数据进行可视化
        if use_env_data:
            temp_data, humidity_data = extract_environmental_data(file_path)
            
            # 对找到的每个潜空间变量进行可视化
            for var_name, tensor in latent_vars.items():
                print(f"  对潜空间变量 '{var_name}' (形状: {tensor.shape}) 使用环境数据进行可视化")
                
                # 处理张量的形状以适应可视化
                original_shape = tensor.shape
                if len(original_shape) > 2:
                    reshaped_data = tensor.reshape(-1, original_shape[-1])
                    print(f"  重塑张量从 {original_shape} 到 {reshaped_data.shape}")
                else:
                    reshaped_data = tensor
                
                # 如果需要进行PCA预处理
                if use_pca_first and reshaped_data.shape[1] > pca_components:
                    print(f"  使用PCA将特征维度从 {reshaped_data.shape[1]} 降至 {pca_components}")
                    pca = PCA(n_components=pca_components)
                    reshaped_data = pca.fit_transform(reshaped_data)
                
                # 进行环境数据3D可视化
                visualize_with_environmental_data(reshaped_data, temp_data, humidity_data, 
                                               file_name, var_name, output_dir, colormap)
        else:
            # 常规t-SNE可视化
            for var_name, tensor in latent_vars.items():
                print(f"  对潜空间变量 '{var_name}' (形状: {tensor.shape}) 进行t-SNE可视化")
                visualize_tensor_tsne(tensor, file_name, var_name, output_dir, 
                                    perplexity, n_clusters, use_pca_first,
                                    pca_components, plot_3d, learning_rate, colormap)
        
        print(f"可视化结果已保存到 {output_dir} 目录")
        return True
            
    except Exception as e:
        print(f"可视化文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def identify_latent_variables(data, file_name):
    """识别数据中的潜空间变量"""
    latent_vars = {}
    
    # 如果是张量，直接处理
    if isinstance(data, torch.Tensor):
        latent_vars["latent"] = data.detach().cpu().numpy()
        return latent_vars
    
    # 如果是字典，寻找潜空间变量
    if isinstance(data, dict):
        # 潜空间变量的常见名称
        latent_keywords = ['latent', 'embedding', 'hidden', 'code', 'z', 'representation', 'features', 'feat']
        
        # 首先尝试精确匹配
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if any(keyword in key.lower() for keyword in latent_keywords):
                    latent_vars[key] = value.detach().cpu().numpy()
        
        # 如果没有找到，检查其他可能的张量
        if not latent_vars:
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 1:
                    # 只添加形状合适的张量
                    # 潜变量通常是2D张量 [batch_size, latent_dim] 或 3D张量 [batch_size, seq_len, latent_dim]
                    if len(value.shape) >= 2:
                        latent_vars[key] = value.detach().cpu().numpy()
    
    # 如果找不到任何潜变量，可能存储在嵌套结构中
    if not latent_vars and isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                nested_vars = identify_latent_variables(value, file_name)
                for nested_key, nested_val in nested_vars.items():
                    latent_vars[f"{key}.{nested_key}"] = nested_val
    
    return latent_vars

def visualize_tensor_tsne(tensor_data, file_name, var_name, output_dir, 
                          perplexity=30, n_clusters=0, use_pca_first=False,
                          pca_components=50, plot_3d=False, learning_rate='auto',
                          colormap='viridis'):
    """使用t-SNE可视化张量数据"""
    # 确保数据是numpy数组
    if isinstance(tensor_data, torch.Tensor):
        tensor_data = tensor_data.detach().cpu().numpy()
    
    # 处理张量的形状以适应t-SNE
    original_shape = tensor_data.shape
    
    # 如果是3D或更高维张量，需要重塑
    if len(original_shape) > 2:
        reshaped_data = tensor_data.reshape(-1, original_shape[-1])
        print(f"  重塑张量从 {original_shape} 到 {reshaped_data.shape} 用于t-SNE")
    else:
        reshaped_data = tensor_data
    
    # 如果数据点太多，采样
    max_points = 5000  # 可以处理更多点
    if reshaped_data.shape[0] > max_points:
        indices = np.random.choice(reshaped_data.shape[0], max_points, replace=False)
        reshaped_data = reshaped_data[indices]
        print(f"  采样 {max_points} 个点用于t-SNE (原始: {tensor_data.shape[0]})")
    
    # 使用PCA预处理降维（可选）
    if use_pca_first and reshaped_data.shape[1] > pca_components:
        from sklearn.decomposition import PCA
        print(f"  使用PCA将特征维度从 {reshaped_data.shape[1]} 降至 {pca_components}")
        pca = PCA(n_components=pca_components)
        reshaped_data = pca.fit_transform(reshaped_data)
        # 保存PCA的解释方差比
        explained_variance = pca.explained_variance_ratio_
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('PCA组件数')
        plt.ylabel('累计解释方差比')
        plt.title(f'PCA解释方差: {var_name}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_pca_variance.png"))
        plt.close()
    
    # 执行t-SNE
    target_dims = 3 if plot_3d else 2
    print(f"  执行t-SNE (perplexity={perplexity}, dims={target_dims})...")
    
    try:
        # 使用更快的算法和较少的迭代
        tsne = TSNE(n_components=target_dims, perplexity=perplexity, n_iter=750, 
                    random_state=42, method='barnes_hut', learning_rate=learning_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne_result = tsne.fit_transform(reshaped_data)
    except Exception as e:
        print(f"  t-SNE计算出错: {e}")
        print("  尝试PCA替代方案...")
        # 如果t-SNE失败，回退到PCA
        pca = PCA(n_components=target_dims)
        tsne_result = pca.fit_transform(reshaped_data)
    
    # 可视化t-SNE结果
    if plot_3d:
        visualize_3d_tsne(tsne_result, file_name, var_name, output_dir, n_clusters, colormap)
    else:
        visualize_2d_tsne(tsne_result, file_name, var_name, output_dir, n_clusters, colormap)
    
    # 如果需要聚类
    if n_clusters > 0:
        perform_clustering(reshaped_data, tsne_result, n_clusters, file_name, var_name, output_dir)

def visualize_2d_tsne(tsne_result, file_name, var_name, output_dir, n_clusters=0, colormap='viridis'):
    """创建2D t-SNE可视化"""
    plt.figure(figsize=(12, 10))
    
    # 是否进行聚类
    if n_clusters > 0:
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tsne_result)
        
        # 使用聚类结果为点着色
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                     c=clusters, cmap=colormap, alpha=0.6, s=10)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"t-SNE Visualization with Clustering (k={n_clusters}): {var_name}")
    else:
        # 使用点的密度为点着色
        from scipy.stats import gaussian_kde
        xy = np.vstack([tsne_result[:, 0], tsne_result[:, 1]])
        try:
            z = gaussian_kde(xy)(xy)
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         c=z, cmap=colormap, alpha=0.6, s=10)
            plt.colorbar(scatter, label='Density')
        except:
            # 如果密度计算失败，退回到无色散点图
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         alpha=0.6, s=10)
        plt.title(f"t-SNE Visualization: {var_name}")
    
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    
    # 保存文件名中添加聚类信息
    if n_clusters > 0:
        plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_tsne_cluster{n_clusters}.png"))
    else:
        plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_tsne.png"))
    plt.close()

def visualize_3d_tsne(tsne_result, file_name, var_name, output_dir, n_clusters=0, colormap='viridis'):
    """创建3D t-SNE可视化"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 是否进行聚类
    if n_clusters > 0:
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tsne_result)
        
        # 使用聚类结果为点着色
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2],
                    c=clusters, cmap=colormap, alpha=0.6, s=10)
        plt.colorbar(scatter, label='Cluster')
        ax.set_title(f"3D t-SNE with Clustering (k={n_clusters}): {var_name}")
    else:
        # 使用Z坐标为点着色
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2],
                    c=tsne_result[:, 2], cmap=colormap, alpha=0.6, s=10)
        plt.colorbar(scatter, label='t-SNE dim 3')
        ax.set_title(f"3D t-SNE Visualization: {var_name}")
    
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_zlabel("t-SNE dimension 3")
    plt.tight_layout()
    
    # 保存不同角度的图
    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    
    # 基本文件名
    if n_clusters > 0:
        base_filename = os.path.join(output_dir, f"{file_name}_{var_name}_tsne3d_cluster{n_clusters}")
    else:
        base_filename = os.path.join(output_dir, f"{file_name}_{var_name}_tsne3d")
    
    # 保存不同视角的图
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev, azim)
        plt.savefig(f"{base_filename}_view{i+1}.png")
    
    plt.close()

def perform_clustering(original_data, tsne_data, n_clusters, file_name, var_name, output_dir):
    """进行聚类分析并可视化"""
    # 使用KMeans对原始高维数据进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(original_data)
    
    # 计算簇中心
    cluster_centers = kmeans.cluster_centers_
    
    # 分析每个簇的大小
    cluster_sizes = np.bincount(clusters)
    
    # 绘制簇大小分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), cluster_sizes)
    plt.xlabel('簇索引')
    plt.ylabel('簇中点的数量')
    plt.title(f'聚类大小分布: {var_name} (k={n_clusters})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_cluster_sizes_k{n_clusters}.png"))
    plt.close()
    
    # 执行PCA获取主要变化方向
    if original_data.shape[1] > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(original_data)
        
        # 绘制PCA降维结果，使用聚类着色
        plt.figure(figsize=(12, 10))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(label='Cluster')
        plt.title(f"PCA Visualization with Clustering (k={n_clusters}): {var_name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_pca_cluster{n_clusters}.png"))
        plt.close()

def extract_environmental_data(file_path):
    """尝试从同一目录中提取温度和湿度数据"""
    # 首先尝试从文件本身提取
    try:
        data = torch.load(file_path)
        if isinstance(data, dict):
            temp = None
            humidity = None
            
            # 直接在文件中查找温度和湿度
            if 'temperature' in data and data['temperature'] is not None:
                temp = data['temperature']
                if isinstance(temp, torch.Tensor):
                    temp = temp.detach().cpu().numpy()
                print("直接从文件中提取温度数据")
            
            if 'humidity' in data and data['humidity'] is not None:
                humidity = data['humidity']
                if isinstance(humidity, torch.Tensor):
                    humidity = humidity.detach().cpu().numpy()
                print("直接从文件中提取湿度数据")
            
            if temp is not None and humidity is not None:
                print(f"温度数据形状: {temp.shape}, 湿度数据形状: {humidity.shape}")
                return temp, humidity
    except Exception as e:
        print(f"从文件中提取环境数据时出错: {e}")
    
    # 获取目录和基本文件名
    directory = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    
    # 寻找可能的环境数据文件
    env_files = [
        os.path.join(directory, "environmental_data.pt"),
        os.path.join(directory, f"{file_name}_env.pt"),
        os.path.join(directory, "env_data.pt"),
        os.path.join(directory, "environment.pt")
    ]
    
    # 尝试读取每个可能的文件
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                env_data = torch.load(env_file)
                print(f"找到环境数据文件: {env_file}")
                
                # 尝试提取温度和湿度
                if isinstance(env_data, dict):
                    temp = None
                    humidity = None
                    
                    # 搜索温度相关键
                    for key in ['temperature', 'temp', 'T']:
                        if key in env_data:
                            temp = env_data[key]
                            if isinstance(temp, torch.Tensor):
                                temp = temp.detach().cpu().numpy()
                            break
                    
                    # 搜索湿度相关键
                    for key in ['humidity', 'humid', 'H', 'RH']:
                        if key in env_data:
                            humidity = env_data[key]
                            if isinstance(humidity, torch.Tensor):
                                humidity = humidity.detach().cpu().numpy()
                            break
                    
                    if temp is not None and humidity is not None:
                        return temp, humidity
            except:
                pass
    
    print("未找到有效的环境数据文件，将使用随机数据进行可视化演示")
    # 如果找不到真实数据，创建随机数据用于演示
    # 假设潜变量的第一个维度是样本数
    data = torch.load(file_path)
    if isinstance(data, dict) and 'features' in data:
        n_samples = data['features'].shape[0]
    else:
        n_samples = 100
    
    temp = np.random.uniform(20, 30, n_samples)
    humidity = np.random.uniform(40, 80, n_samples)
    
    return temp, humidity

def visualize_with_environmental_data(latent_data, temp_data, humidity_data, file_name, var_name, output_dir, colormap='viridis'):
    """创建3D散点图，使用温度和湿度作为x和y轴，t-SNE结果作为z轴，使用时间信息进行分析"""
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    
    # 下采样数据以加快处理速度
    if latent_data.shape[0] > 5000:
        print(f"  数据点过多 ({latent_data.shape[0]})，随机采样5000个点以加快处理")
        indices = np.random.choice(latent_data.shape[0], 5000, replace=False)
        latent_data = latent_data[indices]
    
    # 执行t-SNE获得一个维度
    print(f"  对潜空间变量执行一维t-SNE...")
    try:
        tsne = TSNE(n_components=1, perplexity=min(30, latent_data.shape[0]//100), 
                   n_iter=750, random_state=42, method='barnes_hut', verbose=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne_result = tsne.fit_transform(latent_data).flatten()
    except Exception as e:
        print(f"  t-SNE计算出错: {e}")
        print("  尝试PCA替代方案...")
        # 如果t-SNE失败，回退到PCA
        pca = PCA(n_components=1)
        tsne_result = pca.fit_transform(latent_data).flatten()
    
    print(f"  温度数据形状: {temp_data.shape}")
    print(f"  湿度数据形状: {humidity_data.shape}")
    
    # 根据温度数据形状，决定使用哪种方式处理
    if len(temp_data.shape) > 1 and temp_data.shape[1] > 1:
        # 对于每天多个温度值的情况
        # 创建有意义的温度特征：日均温、日温差
        if temp_data.shape[1] >= 24:  # 假设有24小时数据
            daily_mean_temp = np.mean(temp_data, axis=1)
            daily_temp_range = np.max(temp_data, axis=1) - np.min(temp_data, axis=1)
            print(f"  计算日均温和日温差，形状：{daily_mean_temp.shape}")
        else:
            # 如果列数少于24，直接使用第一列作为温度特征
            daily_mean_temp = temp_data[:, 0]
            # 使用最大和最小列之差作为温度范围
            daily_temp_range = np.max(temp_data, axis=1) - np.min(temp_data, axis=1)
    else:
        # 单列温度数据
        daily_mean_temp = temp_data
        # 创建伪温度范围（可以考虑替换为其他特征）
        daily_temp_range = np.zeros_like(temp_data)
    
    # 同样处理湿度数据
    if len(humidity_data.shape) > 1 and humidity_data.shape[1] > 1:
        daily_mean_humidity = np.mean(humidity_data, axis=1)
    else:
        daily_mean_humidity = humidity_data
    
    # 确保数据长度匹配
    n_samples = min(len(tsne_result), len(daily_mean_temp), len(daily_mean_humidity))
    
    if n_samples < 10:
        print("  数据点太少，无法创建有意义的可视化")
        return
    
    print(f"  使用 {n_samples} 个样本进行可视化")
    
    # 截取相同长度的数据
    tsne_result = tsne_result[:n_samples]
    daily_mean_temp = daily_mean_temp[:n_samples]
    daily_temp_range = daily_temp_range[:n_samples]
    daily_mean_humidity = daily_mean_humidity[:n_samples]
    
    # 创建季节/月份颜色映射（模拟数据，实际应该使用时间戳）
    # 假设样本是按月份或季节顺序排列的
    # 创建月份/季节索引（为演示目的，使用样本索引的模运算）
    month_indices = np.arange(n_samples) % 12 + 1  # 1-12表示月份
    season_indices = (np.arange(n_samples) % 12) // 3  # 0-3表示四季
    
    # 为了更好的可视化，使用多种不同的彩色方案
    # 1. 按温度梯度着色
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(daily_mean_temp, daily_mean_humidity, tsne_result, 
                c=daily_mean_temp, cmap='coolwarm', alpha=0.8, s=15)
    plt.colorbar(scatter, label='日均温')
    ax.set_xlabel("日均温度")
    ax.set_ylabel("日均湿度")
    ax.set_zlabel("t-SNE维度")
    ax.set_title(f"按温度着色的3D可视化: {var_name}")
    plt.tight_layout()
    base_filename = os.path.join(output_dir, f"{file_name}_{var_name}_温度湿度_按温度")
    # 保存不同角度的图
    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev, azim)
        plt.savefig(f"{base_filename}_视角{i+1}.png")
    plt.close()
    
    # 2. 按温度范围着色
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(daily_mean_temp, daily_mean_humidity, tsne_result, 
                c=daily_temp_range, cmap='plasma', alpha=0.8, s=15)
    plt.colorbar(scatter, label='日温差')
    ax.set_xlabel("日均温度")
    ax.set_ylabel("日均湿度")
    ax.set_zlabel("t-SNE维度")
    ax.set_title(f"按温度范围着色的3D可视化: {var_name}")
    plt.tight_layout()
    base_filename = os.path.join(output_dir, f"{file_name}_{var_name}_温度湿度_按温差")
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev, azim)
        plt.savefig(f"{base_filename}_视角{i+1}.png")
    plt.close()
    
    # 3. 按季节着色
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    seasons = ['冬', '春', '夏', '秋']
    season_colors = ['#1E88E5', '#4CAF50', '#FFC107', '#E53935']  # 蓝、绿、黄、红
    cmap = colors.ListedColormap(season_colors)
    scatter = ax.scatter(daily_mean_temp, daily_mean_humidity, tsne_result, 
                c=season_indices, cmap=cmap, alpha=0.8, s=15)
    # 自定义季节颜色条
    cbar = plt.colorbar(scatter, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(seasons)
    cbar.set_label('季节')
    ax.set_xlabel("日均温度")
    ax.set_ylabel("日均湿度")
    ax.set_zlabel("t-SNE维度")
    ax.set_title(f"按季节着色的3D可视化: {var_name}")
    plt.tight_layout()
    base_filename = os.path.join(output_dir, f"{file_name}_{var_name}_温度湿度_按季节")
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev, azim)
        plt.savefig(f"{base_filename}_视角{i+1}.png")
    plt.close()
    
    # 4. 创建通过温度-湿度散点图展示聚类模式
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(daily_mean_temp, daily_mean_humidity, c=tsne_result, 
                      cmap='viridis', alpha=0.7, s=25)
    plt.colorbar(scatter, label='t-SNE值')
    plt.xlabel('日均温度')
    plt.ylabel('日均湿度')
    plt.title(f'温度-湿度关系 (t-SNE着色): {var_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_温度湿度_散点图.png"))
    plt.close()
    
    # 5. K-means聚类分析
    try:
        # 创建用于聚类的特征矩阵
        feature_matrix = np.column_stack([daily_mean_temp, daily_mean_humidity, tsne_result])
        
        # 选择合适的聚类数量（可以尝试不同的值）
        n_clusters = 4
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # 绘制聚类结果
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用不同的颜色表示不同的聚类
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax.scatter(daily_mean_temp[mask], 
                      daily_mean_humidity[mask], 
                      tsne_result[mask], 
                      color=cluster_colors[i], 
                      label=f'聚类 {i+1}',
                      alpha=0.8,
                      s=25)
        
        ax.set_xlabel('日均温度')
        ax.set_ylabel('日均湿度')
        ax.set_zlabel('t-SNE维度')
        ax.set_title(f'温度-湿度-t-SNE空间的聚类分析: {var_name}')
        ax.legend()
        
        base_filename = os.path.join(output_dir, f"{file_name}_{var_name}_聚类分析")
        for i, (elev, azim) in enumerate(angles):
            ax.view_init(elev, azim)
            plt.savefig(f"{base_filename}_视角{i+1}.png")
        plt.close()
        
        # 也创建一个2D聚类视图
        plt.figure(figsize=(12, 10))
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(daily_mean_temp[mask], 
                       daily_mean_humidity[mask],
                       color=cluster_colors[i],
                       label=f'聚类 {i+1}',
                       alpha=0.7,
                       s=25)
        
        plt.xlabel('日均温度')
        plt.ylabel('日均湿度')
        plt.title(f'温度-湿度空间的聚类分析: {var_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{file_name}_{var_name}_温度湿度_聚类.png"))
        plt.close()
        
    except Exception as e:
        print(f"  执行聚类分析时出错: {e}")

def process_directory(directory, output_dir='./latent_visualizations', 
                      perplexity=30, n_clusters=0, use_pca_first=False,
                      pca_components=50, plot_3d=False, learning_rate='auto',
                      colormap='viridis', specific_key=None, use_env_data=False):
    """处理目录中的所有.pt文件"""
    pt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    
    if not pt_files:
        print(f"在目录 {directory} 中没有找到.pt文件")
        return
    
    print(f"找到 {len(pt_files)} 个.pt文件，开始处理...")
    
    success_count = 0
    for idx, file_path in enumerate(pt_files):
        print(f"[{idx+1}/{len(pt_files)}] 处理文件: {file_path}")
        
        # 为每个文件创建单独的输出子目录
        file_name = os.path.basename(file_path).split('.')[0]
        file_output_dir = os.path.join(output_dir, file_name)
        
        if visualize_latent_space(file_path, file_output_dir, perplexity, n_clusters, 
                                  use_pca_first, pca_components, plot_3d, 
                                  learning_rate, colormap, specific_key, use_env_data):
            success_count += 1
    
    print(f"\n处理完成! 成功可视化 {success_count}/{len(pt_files)} 个文件")
    print(f"所有可视化结果已保存到 {output_dir} 目录的子文件夹中")

def main():
    parser = argparse.ArgumentParser(description='t-SNE可视化.pt文件中的潜空间变量')
    # 路径参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('path', type=str, nargs='?', help='.pt文件或包含.pt文件的目录的路径')
    group.add_argument('--path', type=str, dest='path_option', help='.pt文件或包含.pt文件的目录的路径')
    
    # 输出选项
    parser.add_argument('--output', type=str, default='./latent_visualizations', help='输出可视化图像的目录')
    parser.add_argument('--save-in-place', action='store_true', help='将可视化结果保存在输入路径下，而不是创建新的输出目录')
    
    # 数据选择
    parser.add_argument('--key', type=str, help='指定要可视化的键名，适用于知道具体键名的情况')
    
    # t-SNE参数
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE的复杂度参数')
    parser.add_argument('--learning-rate', type=str, default='auto', help='t-SNE的学习率, 可以是"auto"或数值')
    
    # 预处理和可视化选项
    parser.add_argument('--clusters', type=int, default=0, help='使用KMeans进行聚类，指定簇的数量 (0表示不聚类)')
    parser.add_argument('--use-pca', action='store_true', help='在t-SNE前先使用PCA降维')
    parser.add_argument('--pca-components', type=int, default=50, help='PCA降维的目标维度')
    parser.add_argument('--3d', action='store_true', dest='plot_3d', help='生成3D t-SNE可视化')
    parser.add_argument('--use-env-data', action='store_true', help='使用温度和湿度作为坐标轴，进行3D可视化')
    parser.add_argument('--colormap', type=str, default='viridis', help='可视化使用的颜色映射')
    
    args = parser.parse_args()
    
    # 使用提供的路径(位置参数或选项参数)
    target_path = args.path if args.path else args.path_option
    
    # 处理学习率参数
    learning_rate = args.learning_rate
    if learning_rate != 'auto':
        try:
            learning_rate = float(learning_rate)
        except ValueError:
            print(f"警告: 无效的学习率 '{learning_rate}'，使用'auto'替代")
            learning_rate = 'auto'
    
    # 如果选择了原地保存，则将输出目录设置为输入路径
    if args.save_in_place:
        if os.path.isdir(target_path):
            output_dir = target_path
            print(f"结果将保存在输入目录中: {output_dir}")
        else:
            # 如果是单个文件，则使用其所在目录
            output_dir = os.path.dirname(target_path)
            print(f"结果将保存在文件所在目录中: {output_dir}")
    else:
        output_dir = args.output
    
    # 判断输入是文件还是目录
    if os.path.isdir(target_path):
        print(f"处理目录: {target_path}")
        process_directory(target_path, output_dir, args.perplexity, args.clusters, 
                          args.use_pca, args.pca_components, args.plot_3d, 
                          learning_rate, args.colormap, args.key, args.use_env_data)
    else:
        visualize_latent_space(target_path, output_dir, args.perplexity, args.clusters, 
                              args.use_pca, args.pca_components, args.plot_3d, 
                              learning_rate, args.colormap, args.key, args.use_env_data)

if __name__ == "__main__":
    main() 