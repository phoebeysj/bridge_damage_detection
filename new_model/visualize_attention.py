import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import glob

def load_attention_weights(file_path):
    """
    加载注意力权重文件
    
    Args:
        file_path: 注意力权重文件路径
    
    Returns:
        加载的数据字典
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        data = torch.load(file_path)
        print(f"成功加载 {file_path}")
        print(f"数据包含以下键: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

def visualize_attention_heatmap(attention_data, save_path=None, max_heads=3, max_samples=5):
    """
    可视化注意力热力图
    
    Args:
        attention_data: 包含注意力权重的字典
        save_path: 保存路径，如果为None则显示图像
        max_heads: 最大可视化的注意力头数量
        max_samples: 最大可视化的样本数量
    """
    if not attention_data:
        print("没有提供注意力数据")
        return
    
    # 确定注意力权重键
    attn_keys = [k for k in attention_data.keys() if isinstance(attention_data[k], torch.Tensor) and 
                len(attention_data[k].shape) >= 3]
    
    if not attn_keys:
        print("数据中没有找到注意力权重张量")
        return
    
    # 获取温度信息
    temp_info = ""
    if 'temperature' in attention_data:
        temps = attention_data['temperature']
        temp_info = f", 温度范围: {np.min(temps):.1f}°C-{np.max(temps):.1f}°C"
    
    if 'temp_range' in attention_data:
        temp_info = f", 温度区间: {attention_data['temp_range'][0]}-{attention_data['temp_range'][1]}°C"
    
    # 处理每个注意力键
    for attn_key in attn_keys:
        attention = attention_data[attn_key]
        
        # 处理形状和维度
        if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attention.shape
            
            # 限制样本数量和头数
            num_samples = min(batch_size, max_samples)
            num_heads_to_plot = min(num_heads, max_heads)
            
            # 创建子图
            fig, axes = plt.subplots(num_samples, num_heads_to_plot, 
                                     figsize=(4*num_heads_to_plot, 3*num_samples),
                                     squeeze=False)
            fig.suptitle(f"注意力热力图 - {attn_key}{temp_info}", fontsize=16)
            
            # 绘制热力图
            for i in range(num_samples):
                for j in range(num_heads_to_plot):
                    ax = axes[i, j]
                    attn_map = attention[i, j].numpy()
                    sns.heatmap(attn_map, ax=ax, cmap='viridis', vmin=0, vmax=1)
                    if i == 0:
                        ax.set_title(f"头 {j+1}")
                    if j == 0:
                        ax.set_ylabel(f"样本 {i+1}")
                    ax.set_xlabel("目标位置")
                    ax.set_ylabel("源位置")
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_path:
                plt.savefig(os.path.join(save_path, f"attn_heatmap_{attn_key}.png"), dpi=300)
                plt.close()
            else:
                plt.show()
        
        elif len(attention.shape) == 3:  # [batch, seq_len, seq_len]
            batch_size, seq_len, _ = attention.shape
            num_samples = min(batch_size, max_samples)
            
            # 创建子图
            fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4), squeeze=False)
            fig.suptitle(f"注意力热力图 - {attn_key}{temp_info}", fontsize=16)
            
            # 绘制热力图
            for i in range(num_samples):
                ax = axes[0, i]
                attn_map = attention[i].numpy()
                sns.heatmap(attn_map, ax=ax, cmap='viridis')
                ax.set_title(f"样本 {i+1}")
                ax.set_xlabel("目标位置")
                ax.set_ylabel("源位置")
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_path:
                plt.savefig(os.path.join(save_path, f"attn_heatmap_{attn_key}.png"), dpi=300)
                plt.close()
            else:
                plt.show()

def visualize_latent_features(features_data, save_path=None, method='tsne'):
    """
    可视化潜空间特征
    
    Args:
        features_data: 包含潜空间特征的字典
        save_path: 保存路径，如果为None则显示图像
        method: 降维方法，'tsne'或'pca'
    """
    if not features_data or 'features' not in features_data:
        print("没有提供有效的特征数据")
        return
    
    # 获取特征
    features = features_data['features']
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    # 重塑特征为2D，如果需要
    if len(features.shape) > 2:
        batch_size = features.shape[0]
        features = features.reshape(batch_size, -1)  # 将所有维度展平为每个样本一行
    
    # 颜色映射
    colors = None
    color_label = None
    
    # 检查是否有温度数据用于颜色编码
    if 'temperature' in features_data:
        colors = features_data['temperature']
        color_label = '温度 (°C)'
    
    # 使用t-SNE或PCA进行降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        title = "t-SNE 潜空间特征可视化"
    else:  # pca
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)
        title = "PCA 潜空间特征可视化"
    
    # 创建数据框用于绘图
    df = pd.DataFrame({
        'x': reduced_features[:, 0],
        'y': reduced_features[:, 1]
    })
    
    if colors is not None:
        df['color'] = colors
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    if colors is not None:
        scatter = plt.scatter(df['x'], df['y'], c=df['color'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=color_label)
    else:
        plt.scatter(df['x'], df['y'], alpha=0.7)
    
    # 添加标题和标签
    dataset_type = features_data.get('dataset_type', '')
    temp_range = features_data.get('temp_range', '')
    
    if dataset_type:
        title += f" - {dataset_type}"
    if temp_range:
        title += f" (温度区间: {temp_range[0]}-{temp_range[1]}°C)"
    
    plt.title(title)
    plt.xlabel("维度 1")
    plt.ylabel("维度 2")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f"latent_features_{method}_{dataset_type}.png"), dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_temperature_trends(dir_path, pattern="*_temp_*_*_*.pt"):
    """
    可视化不同温度区间下的注意力模式变化
    
    Args:
        dir_path: 包含注意力权重文件的目录
        pattern: 文件匹配模式
    """
    # 查找所有温度相关的注意力文件
    file_paths = glob.glob(os.path.join(dir_path, pattern))
    file_paths.sort()
    
    if not file_paths:
        print(f"在 {dir_path} 中没有找到匹配 {pattern} 的文件")
        return
    
    # 提取温度区间和加载数据
    temp_bins = []
    avg_attentions = []
    attentions_by_key = {}
    
    for file_path in file_paths:
        try:
            data = torch.load(file_path)
            if 'temp_range' in data:
                temp_bin = data['temp_range']
                temp_bins.append(f"{temp_bin[0]}-{temp_bin[1]}°C")
                
                # 查找所有注意力键
                attn_keys = [k for k in data.keys() if isinstance(data[k], torch.Tensor) and 
                            len(data[k].shape) >= 3]
                
                # 为每个键计算平均注意力
                for key in attn_keys:
                    if key not in attentions_by_key:
                        attentions_by_key[key] = []
                    
                    attn = data[key]
                    if len(attn.shape) == 4:  # [batch, heads, seq, seq]
                        avg_attn = attn.mean(dim=(0, 1))  # 在批次和头上平均
                    elif len(attn.shape) == 3:  # [batch, seq, seq]
                        avg_attn = attn.mean(dim=0)  # 在批次上平均
                    
                    attentions_by_key[key].append(avg_attn)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 按照温度区间对每个键的注意力进行可视化
    for key, attentions in attentions_by_key.items():
        if not attentions:
            continue
        
        num_bins = len(temp_bins)
        fig, axes = plt.subplots(1, num_bins, figsize=(5*num_bins, 5), squeeze=False)
        fig.suptitle(f"不同温度区间的注意力热力图 - {key}", fontsize=16)
        
        for i, (temp_bin, attn) in enumerate(zip(temp_bins, attentions)):
            ax = axes[0, i]
            sns.heatmap(attn.numpy(), ax=ax, cmap='viridis')
            ax.set_title(f"温度: {temp_bin}")
            ax.set_xlabel("目标位置")
            ax.set_ylabel("源位置")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(dir_path, f"temp_trends_{key}.png"), dpi=300)
        plt.close()

def analyze_global_features(file_path):
    """
    分析和可视化全局特征
    
    Args:
        file_path: 全局特征文件路径
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        data = torch.load(file_path)
        print(f"成功加载全局特征数据，内含键: {list(data.keys())}")
        
        # 检查必要的键
        required_keys = ['features', 'temperature']
        if not all(k in data for k in required_keys):
            print(f"数据缺少必要的键: {[k for k in required_keys if k not in data]}")
            return
        
        features = data['features']
        temps = data['temperature']
        
        # PCA 可视化
        visualize_latent_features(data, save_path=os.path.dirname(file_path), method='pca')
        
        # t-SNE 可视化
        visualize_latent_features(data, save_path=os.path.dirname(file_path), method='tsne')
        
        # 特征-温度相关性分析
        if len(features.shape) > 2:
            batch_size = features.shape[0]
            features_flat = features.reshape(batch_size, -1)
        else:
            features_flat = features
        
        # 计算每个特征维度与温度的相关性
        correlations = []
        for i in range(min(20, features_flat.shape[1])):  # 只取前20个维度避免过多计算
            corr = np.corrcoef(features_flat[:, i].numpy(), temps)[0, 1]
            correlations.append((i, corr))
        
        # 按相关性排序
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 绘制前5个相关性最强的特征与温度的散点图
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, (dim, corr) in enumerate(correlations[:5]):
            ax = axes[i]
            ax.scatter(temps, features_flat[:, dim].numpy(), alpha=0.5)
            ax.set_title(f"维度 {dim}: 相关性 = {corr:.2f}")
            ax.set_xlabel("温度 (°C)")
            ax.set_ylabel(f"特征值 {dim}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(file_path), "feature_temp_correlation.png"), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"处理全局特征时出错: {e}")

def compare_attention_types(file_path, attention_types=None, save_path=None, max_samples=3):
    """
    比较不同类型的注意力权重（如时间、空间、环境注意力）
    
    Args:
        file_path: 注意力权重文件路径
        attention_types: 要比较的注意力类型列表，None则自动检测
        save_path: 保存路径，如果为None则显示图像
        max_samples: 最大可视化的样本数量
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        data = torch.load(file_path)
        print(f"成功加载 {file_path}")
        print(f"数据包含以下键: {list(data.keys())}")
        
        # 确定注意力权重键
        attn_keys = [k for k in data.keys() if isinstance(data[k], torch.Tensor) and 
                     len(data[k].shape) >= 3]
        
        if not attn_keys:
            print("数据中没有找到注意力权重张量")
            return
        
        # 如果未指定注意力类型，尝试根据键名推断
        if attention_types is None:
            # 常见注意力类型的关键词
            time_keywords = ['time', 'temporal', 'attn_t']
            space_keywords = ['space', 'spatial', 'attn_s']
            env_keywords = ['env', 'environment', 'attn_e']
            
            time_keys = [k for k in attn_keys if any(kw in k.lower() for kw in time_keywords)]
            space_keys = [k for k in attn_keys if any(kw in k.lower() for kw in space_keywords)]
            env_keys = [k for k in attn_keys if any(kw in k.lower() for kw in env_keywords)]
            
            # 如果没有匹配到任何关键词，使用所有键
            if not (time_keys or space_keys or env_keys):
                attention_types = attn_keys
            else:
                attention_types = []
                if time_keys:
                    attention_types.extend(time_keys)
                if space_keys:
                    attention_types.extend(space_keys)
                if env_keys:
                    attention_types.extend(env_keys)
        
        # 过滤不存在的注意力类型
        attention_types = [t for t in attention_types if t in attn_keys]
        
        if not attention_types:
            print("没有找到指定类型的注意力权重")
            return
        
        # 获取温度信息
        temp_info = ""
        if 'temperature' in data:
            temps = data['temperature']
            temp_info = f", 温度范围: {np.min(temps):.1f}°C-{np.max(temps):.1f}°C"
        
        if 'temp_range' in data:
            temp_info = f", 温度区间: {data['temp_range'][0]}-{data['temp_range'][1]}°C"
        
        # 确定最大样本数
        min_batch_size = float('inf')
        for attn_type in attention_types:
            attn = data[attn_type]
            if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                min_batch_size = min(min_batch_size, attn.shape[0])
            elif len(attn.shape) == 3:  # [batch, seq_len, seq_len]
                min_batch_size = min(min_batch_size, attn.shape[0])
        
        num_samples = min(min_batch_size, max_samples)
        
        # 为每个样本创建比较图
        for sample_idx in range(num_samples):
            fig, axes = plt.subplots(1, len(attention_types), 
                                   figsize=(6*len(attention_types), 5),
                                   squeeze=False)
            fig.suptitle(f"不同类型注意力比较 - 样本 {sample_idx+1}{temp_info}", fontsize=16)
            
            for i, attn_type in enumerate(attention_types):
                ax = axes[0, i]
                attn = data[attn_type]
                
                if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    # 对所有头取平均
                    attn_map = attn[sample_idx].mean(dim=0).numpy()
                elif len(attn.shape) == 3:  # [batch, seq_len, seq_len]
                    attn_map = attn[sample_idx].numpy()
                
                sns.heatmap(attn_map, ax=ax, cmap='viridis')
                ax.set_title(f"{attn_type}")
                ax.set_xlabel("目标位置")
                ax.set_ylabel("源位置")
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_path:
                plt.savefig(os.path.join(save_path, f"attn_compare_sample_{sample_idx+1}.png"), dpi=300)
                plt.close()
            else:
                plt.show()
                
        # 比较不同注意力类型的平均模式
        fig, axes = plt.subplots(1, len(attention_types), 
                               figsize=(6*len(attention_types), 5),
                               squeeze=False)
        fig.suptitle(f"不同类型注意力的平均模式{temp_info}", fontsize=16)
        
        for i, attn_type in enumerate(attention_types):
            ax = axes[0, i]
            attn = data[attn_type]
            
            if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                # 在批次和头上取平均
                attn_map = attn.mean(dim=(0, 1)).numpy()
            elif len(attn.shape) == 3:  # [batch, seq_len, seq_len]
                # 在批次上取平均
                attn_map = attn.mean(dim=0).numpy()
            
            sns.heatmap(attn_map, ax=ax, cmap='viridis')
            ax.set_title(f"{attn_type} - 平均")
            ax.set_xlabel("目标位置")
            ax.set_ylabel("源位置")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(os.path.join(save_path, f"attn_compare_average.png"), dpi=300)
            plt.close()
        else:
            plt.show()
            
        # 如果有多个注意力类型，计算它们之间的相关性
        if len(attention_types) > 1:
            # 为每个注意力类型准备展平的数据
            flattened_attns = []
            for attn_type in attention_types:
                attn = data[attn_type]
                if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    # 在头上取平均后展平
                    flat_attn = attn.mean(dim=1).reshape(attn.shape[0], -1)
                elif len(attn.shape) == 3:  # [batch, seq_len, seq_len]
                    flat_attn = attn.reshape(attn.shape[0], -1)
                flattened_attns.append(flat_attn)
            
            # 计算相关性矩阵
            corr_matrix = np.zeros((len(attention_types), len(attention_types)))
            for i in range(len(attention_types)):
                for j in range(len(attention_types)):
                    # 对每个样本计算相关性，然后取平均
                    sample_corrs = []
                    for s in range(min(flattened_attns[i].shape[0], flattened_attns[j].shape[0])):
                        corr = np.corrcoef(flattened_attns[i][s].numpy(), flattened_attns[j][s].numpy())[0, 1]
                        if not np.isnan(corr):
                            sample_corrs.append(corr)
                    
                    if sample_corrs:
                        corr_matrix[i, j] = np.mean(sample_corrs)
            
            # 绘制相关性热力图
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                       xticklabels=attention_types, yticklabels=attention_types)
            plt.title("不同注意力类型间的相关性")
            
            if save_path:
                plt.savefig(os.path.join(save_path, f"attn_correlation.png"), dpi=300)
                plt.close()
            else:
                plt.show()
    
    except Exception as e:
        print(f"处理注意力比较时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数，用于演示如何使用上述函数"""
    # 示例路径 - 需要根据实际情况修改
    base_path = "./checkpoints/your_experiment_name"
    
    # 1. 读取和可视化注意力权重
    attn_path = os.path.join(base_path, "vali/attention_weights")
    if os.path.exists(attn_path):
        attn_files = glob.glob(os.path.join(attn_path, "*.pt"))
        if attn_files:
            print(f"找到 {len(attn_files)} 个注意力权重文件")
            for file in attn_files[:3]:  # 只处理前3个文件作为示例
                data = load_attention_weights(file)
                if data:
                    visualize_attention_heatmap(data, save_path=attn_path)
    
    # 2. 可视化温度相关的注意力趋势
    temp_attn_path = os.path.join(base_path, "vali/attention_weights_min_temp")
    if os.path.exists(temp_attn_path):
        visualize_temperature_trends(temp_attn_path)
    
    # 3. 读取和可视化潜空间特征
    feature_path = os.path.join(base_path, "vali/latent_features")
    if os.path.exists(feature_path):
        feature_files = glob.glob(os.path.join(feature_path, "*.pt"))
        if feature_files:
            print(f"找到 {len(feature_files)} 个潜空间特征文件")
            for file in feature_files:
                analyze_global_features(file)

if __name__ == "__main__":
    main() 