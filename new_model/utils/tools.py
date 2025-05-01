import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_mean = None
        self.best_std = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_std = np.inf
        self.delta = delta
        self.best_vali = []
        # self.best_test = []
        # self.best_test_health = []

    def __call__(self, val_loss_mean, val_std, model, path, vali_loss):
        score = -val_std
        improved = False
        if self.best_score is None:
            self.best_score = score
            self.best_mean = val_loss_mean
            self.best_std = val_std
            self.best_vali = vali_loss
            # self.best_test = test_loss
            # self.best_test_health = test_health_loss
            self.save_checkpoint(val_loss_mean, val_std, model, path)
            improved = True
        elif float(score.mean()) < float(self.best_score.mean()) + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_mean = val_loss_mean
            self.best_std = val_std
            self.best_vali = vali_loss
            # self.best_test = test_loss
            # self.best_test_health = test_health_loss
            self.save_checkpoint(val_loss_mean, val_std, model, path)
            self.counter = 0
            improved = True
        return self.best_mean, self.best_std, improved

    def save_checkpoint(self, val_loss, val_std, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_std:} --> {val_std:}).  Saving model ...')
        checkpoint = {
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_std':val_std,
        }
        torch.save(checkpoint, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
        self.val_std = val_std


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(trues, preds=None, path=None):
    """
    Results visualization
    """
    # plt.figure()
    # plt.plot(true, label='GroundTruth',  linestyle='--',marker='', linewidth=0.5)
    # if preds is not None:
    #     plt.plot(preds, label='Prediction', linestyle='-', marker='', linewidth=0.5)
    # plt.legend()
    # plt.savefig(os.join(path, ), bbox_inches='tight')
    plt.style.use('bmh')
    num_sensors = trues.shape[1]
    
    # 创建多个子图，每个子图对应一个传感器
    fig, axes = plt.subplots(num_sensors, 1, figsize=(12, num_sensors * 3), sharex=True)
    
    # 如果只有一个传感器，axes 不是列表，转换成列表以便统一处理
    if num_sensors == 1:
        axes = [axes]
        
    for i in range(num_sensors):
        # 绘制真实值，采用虚线，线宽0.5
        axes[i].plot(trues[:, i], label='GroundTruth', linestyle='--', marker='', linewidth=0.5, color = 'brown')
        # 如果预测值不为 None，则绘制预测值，采用实线，线宽0.5
        if preds is not None:
            axes[i].plot(preds[:, i], label='Prediction', linestyle='-', marker='', linewidth=0.5, color = 'cyan')
        axes[i].legend()
        axes[i].set_title(f'Sensor {i+1}')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Time Index')
    plt.tight_layout()
    
    # 构造保存路径，保存为 'comparison.png'
    save_path = os.path.join(path, "True VS Pred.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 示例调用：
# 假设 true 和 preds 的形状为 (10000, 8)，save_dir 为你想保存图片的目录
# visual_multisensor(true, preds, "/your/save/directory")



def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def inject_sensor_fault(batch_data, fault_ratios, device, time_dim=-1, **kwargs):
    """
    多类型传感器故障注入函数
    :param batch_data: 原始数据 [batch_size, seq_len] 或 [batch_size, channels, seq_len]
    :param fault_ratios: 故障比例配置字典，例如：
        {
            'normal': 0.75,   # 正常数据比例
            'missing': 0.07, # 数据丢失
            'minor': 0.05,   # 振幅衰减
            'square': 0.03,  # 方波模式
            'trend': 0.05,   # 趋势漂移
            'spike': 0.05,   # 尖峰脉冲
            'outlier': 0.03, # 连续离群值
            # 'jump': 0.02,    # 跳跃恢复
        }
    :param time_dim: 时间维度位置
    :param kwargs: 各故障类型参数配置
    :return: 注入故障后的数据
    """
    assert abs(sum(fault_ratios.values()) - 1.0)< 1e-6, "故障比例总和必须为1"
    original_shape = batch_data.shape
    batch_data = batch_data.clone().to(device)
    
    # 将数据统一处理为 [batch_size, seq_len] 格式
    if batch_data.dim() == 3:
        batch_data = batch_data.view(batch_data.size(0), -1)
    batch_size, seq_len = batch_data.shape
    
    # 生成故障类型标签
    fault_types = list(fault_ratios.keys())
    ratios = list(fault_ratios.values())
    indices = np.random.choice(fault_types, size=batch_size, p=ratios)
    default_params = {
    # 缺失值参数
    'missing_mode': 'mean',      # 可选: fixed/mean/median
    'missing_ratio': 0.0,        # 当mode=fixed时使用绝对值，否则为比例
    
    # 振幅衰减参数
    'minor_mode': 'dynamic',     # dynamic/fixed
    'minor_range': (0.3, 0.7),   # 比例范围或固定衰减系数
    
    # 方波参数
    'square_period_ratio': 0.2,  # 周期长度占序列比例
    'square_amp_ratio': 1.0,     # 方波振幅相对信号强度的比例
    'square_duty_ratio': 0.5,    # 方波占空比
    
    # 趋势漂移参数
    'trend_mode': 'std_based',   # std_based/fixed
    'trend_range': (0.01, 0.05), # 当mode=std_based时为std的倍数，否则为固定斜率
    
    # 尖峰参数
    'spike_ratio': 1.5,          # 尖峰强度比例（相对于信号std）
    'spike_width_ratio': 0.01,   # 尖峰持续时间占序列长度的比例
    'spike_count': 3,            # 最大尖峰数量
    
    # 离群值参数
    'outlier_std_ratio': 2.0,    # 离群值强度比例（相对于信号std）
    'outlier_length_ratio':0.001,
    
    # # 跳跃参数
    # 'jump_ratio': 2.0,           # 跳跃幅度比例（相对于信号std）
    # 'jump_duration_ratio': 0.05,  # 跳跃持续时间比例
    # 'jump_recovery': True        # 是否恢复原始值
}
    params = {**default_params, **kwargs}

    # 故障注入主逻辑（以趋势漂移为例）
    for i in range(batch_size):
        data = batch_data[i]
        fault_type = indices[i]
        signal_std = data.std().to(device)
        signal_abs_max = data.abs().max().to(device)
        
        if fault_type == 'normal':
            continue  # 保持原始数据
            
        # 缺失值故障
        elif fault_type == 'missing':
            if params['missing_mode'] == 'mean':
                val = data.mean() * params['missing_ratio']
            elif params['missing_mode'] == 'median':
                val = torch.median(data) * params['missing_ratio']
            else:
                val = params['missing_ratio']
            batch_data[i].fill_(val)

        # 振幅衰减故障
        elif fault_type == 'minor':
            if params['minor_mode'] == 'dynamic':
                min_a = params['minor_range'][0] * (signal_abs_max / data.abs().mean())
                max_a = params['minor_range'][1] * (signal_abs_max / data.abs().mean())
                alpha = torch.empty(1).uniform_(min_a, max_a).to(device)
            else:
                alpha = torch.empty(1).uniform_(*params['minor_range']).to(device)
            batch_data[i] *= alpha.clamp(0.01, 0.99)  # 防止完全归零

        # 方波故障
        elif fault_type == 'square':
            # 计算周期参数
            period = max(10, int(seq_len * params['square_period_ratio']))
            duty = int(period * params['square_duty_ratio'])
            amp = params['square_amp_ratio'] * signal_std
            
            # 生成方波模板
            square_wave = torch.cat([
                torch.ones(duty).to(device) * amp,
                torch.zeros(period - duty).to(device) * amp
            ])
            
            # 扩展至序列长度
            repeats = seq_len // period + 2
            square_wave = torch.tile(square_wave, (repeats,))[:seq_len]
            batch_data[i] += square_wave * np.random.choice([-1, 1])

        # 趋势漂移故障
        elif fault_type == 'trend':
            if params['trend_mode'] == 'std_based':
                min_slope = params['trend_range'][0] * signal_std
                max_slope = params['trend_range'][1] * signal_std
            else:
                min_slope, max_slope = params['trend_range']
                
            slope = torch.empty(1).uniform_(min_slope, max_slope).to(device)
            time = torch.arange(seq_len).to(device)
            batch_data[i] += slope * time

        # 尖峰故障
        elif fault_type == 'spike':
            num_spikes = np.random.randint(1, params['spike_count'])
            width = max(1, int(seq_len * params['spike_width_ratio']))
            
            for _ in range(num_spikes):
                # 动态计算强度
                if isinstance(params['spike_ratio'], (tuple, list)):
                    ratio = np.random.uniform(*params['spike_ratio'])
                else:
                    ratio = params['spike_ratio']
                spike_val = ratio * signal_std
                
                # 随机位置并保持极性
                pos = np.random.randint(width, seq_len-width)
                polarity = torch.sign(data[pos]).to(device)
                batch_data[i, pos:pos+width] += spike_val * polarity

        # 离群值故障
        elif fault_type == 'outlier':
            length = max(3, int(seq_len * params['outlier_length_ratio']))
            start = np.random.randint(0, seq_len-length)
            
            # 生成高斯噪声
            noise = (torch.randn(length) * params['outlier_std_ratio']).to(device)* signal_std
            batch_data[i, start:start+length] += noise

        # # 跳跃故障
        # elif fault_type == 'jump':
        #     duration = max(2, int(seq_len * params['jump_duration_ratio']))
        #     pos = np.random.randint(0, seq_len-duration)
        #     jump_val = params['jump_ratio'] * signal_std
            
        #     # 记录原始值用于恢复
        #     original = data[pos:pos+duration].clone()
        #     batch_data[i, pos:pos+duration] += jump_val * torch.sign(original)
            
        #     # 恢复机制
        #     if params['jump_recovery'] and (pos + 2*duration) < seq_len:
        #         recovery = torch.linspace(1.0, 0.0, duration).to(device)
        #         batch_data[i, pos+duration:pos+2*duration] += jump_val * recovery

    # 恢复原始形状
    if len(original_shape) == 3:
        batch_data = batch_data.view(original_shape)
        
    return batch_data

def diversity_loss(memory):
    norm_mem = F.normalize(memory, p=2, dim=1)  # L2归一化
    similarity = torch.mm(norm_mem, norm_mem.t())  # [9,9]
    mask = ~torch.eye(9, dtype=torch.bool)  # 排除对角线
    return similarity[mask].mean()  # 非对角元素的平均相似度

def plot_parts(losses, trues, preds, path_loss_save, flag):
    # 假设 losses 是一个一维 tensor，每个元素对应一段 1000 个点的 loss
    segment_length = 60000  # 每段数据点数量
    if flag == 'test':
        loss_idx = torch.argmax(losses).item()  # 找到 loss 最大的段的下标
    elif flag == 'test_health' or flag == 'vali':
        loss_idx = torch.argmin(losses).item()

    # 根据下标计算这一段在整个数据中的起始和结束位置
    start_idx = loss_idx * segment_length
    end_idx = (loss_idx + 1) * segment_length

    # 提取对应的 preds 和 trues（假设它们的第一维是时间步）
    segment_trues = trues[start_idx:end_idx, :]
    segment_preds = preds[start_idx:end_idx, :]

    # 调用你已有的 visual 函数进行可视化
    visual(segment_trues, segment_preds, path_loss_save)


def damage_metric(flag, ts, path_loss_save=None):
    # 获取所有标签列（以_label结尾的列）
    label_columns = [col for col in ts.columns if col.endswith('_label') and col != 'true_label']
    
    if not label_columns:
        print(f"警告：在{flag}数据中没有找到标签列！")
        return
    
    # 创建一个字典来存储各传感器的准确率
    accuracy_dict = {}
    
    if flag == 'test':
        # 计算所有传感器标签的平均值作为总体准确率
        print("\n====== 各传感器损伤识别准确率 ======")
        for col in label_columns:
            sensor_accuracy = ts[col].mean()
            sensor_name = col.split('_')[0]
            accuracy_dict[sensor_name] = sensor_accuracy
            print(f"{sensor_name}传感器损伤识别准确率: {sensor_accuracy:.4f}")
        
        # 计算平均准确率
        avg_accuracy = sum(accuracy_dict.values()) / len(accuracy_dict)
        print(f"\n所有传感器损伤测试数据的平均损伤识别准确率为：{avg_accuracy:.4f}")
            
    elif flag == 'test_health':
        # 计算所有传感器标签的平均值作为总体准确率（健康数据期望为0，因此用1减去均值）
        print("\n====== 各传感器健康识别准确率 ======")
        for col in label_columns:
            sensor_accuracy = 1 - ts[col].mean()  # 健康数据应该标签为0，所以用1减去均值
            sensor_name = col.split('_')[0]
            accuracy_dict[sensor_name] = sensor_accuracy
            print(f"{sensor_name}传感器健康识别准确率: {sensor_accuracy:.4f}")
        
        # 计算平均准确率
        avg_accuracy = sum(accuracy_dict.values()) / len(accuracy_dict)
        print(f"\n所有传感器健康测试数据的平均损伤识别准确率为：{avg_accuracy:.4f}")
    
    # 如果提供了保存路径，保存准确率结果
    if path_loss_save:
        # 保存准确率到CSV
        accuracy_df = pd.DataFrame({
            'sensor': list(accuracy_dict.keys()),
            'accuracy': list(accuracy_dict.values())
        })
        accuracy_df.loc[len(accuracy_df)] = ['average', avg_accuracy]  # 添加平均值
        accuracy_path = os.path.join(path_loss_save, f'{flag}_accuracy.csv')
        accuracy_df.to_csv(accuracy_path, index=False)
        print(f"\n准确率已保存至: {accuracy_path}")
        
        # # 创建准确率柱状图
        # plt.figure(figsize=(10, 6))
        # plt.style.use('bmh')
        # bars = plt.bar(accuracy_dict.keys(), accuracy_dict.values(), alpha=0.7)
        # plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'平均值: {avg_accuracy:.4f}')
        
        # # 在柱状图上标注准确率值
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
        #             f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        # plt.xlabel('传感器')
        # plt.ylabel('准确率')
        # plt.title(f'{flag} 数据各传感器准确率')
        # plt.ylim(0, 1.1)  # 设置y轴范围
        # plt.xticks(rotation=45)
        # plt.legend()
        # plt.tight_layout()
        
        # # 保存图表
        # chart_path = os.path.join(path_loss_save, f'{flag}_accuracy_chart.png')
        # plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"准确率可视化已保存至: {chart_path}")
        # summary_dir = os.path.join(path_loss_save, 'accuracy_summary')
        # os.makedirs(summary_dir, exist_ok=True)
        
        # # 生成综合结果摘要文件
        # with open(os.path.join(summary_dir, f'{flag}_summary.txt'), 'w') as f:
        #     f.write("传感器准确率:\n")
        #     for sensor, acc in accuracy_dict.items():
        #         f.write(f"  {sensor}: {acc:.4f}\n")
        #     f.write(f"\n平均准确率: {avg_accuracy:.4f}\n\n")

def result_save(trues, preds, path_loss_save, labels, losses, timestamps, flag, moe_weight=None, sensor_names=None):
    # trues_np = trues.cpu().numpy()
    # preds_np = preds.cpu().numpy()
    
    # # Check if we have multiple sensors (indicated by multi-dimensional data)
    # if len(trues_np.shape) > 1:
    #     # Multi-sensor case: Create a dictionary with columns for each sensor
    #     df_dict = {}
        
    #     # 处理3D张量 (batch_size, seq_len, num_sensors)
    #     if len(trues_np.shape) == 3:
    #         num_sensors = trues_np.shape[2]  # 最后一个维度是传感器数量
    #         for i in range(num_sensors):
    #             # 提取第i个传感器的所有数据并展平
    #             df_dict[f"true_sensor_{i}"] = trues_np[:, :, i].flatten()
    #             df_dict[f"pred_sensor_{i}"] = preds_np[:, :, i].flatten()
    #     # 处理2D张量 (batch_size, num_sensors)
    #     elif trues_np.shape[1] > 1:  # 确保第二个维度大于1，表示有多个传感器
    #         num_sensors = trues_np.shape[1]
    #         for i in [3,5,6,7,10,12,14,16]:
    #             df_dict[f"true_sensor_{i}"] = trues_np[:, i]
    #             df_dict[f"pred_sensor_{i}"] = preds_np[:, i]
    #     else:
    #         # 单传感器的2D情况 (batch_size, 1)
    #         df_dict["true"] = trues_np.flatten()
    #         df_dict["pred"] = preds_np.flatten()
            
    #     df_pred_true = pd.DataFrame(df_dict)
    # else:
    #     # Single sensor case (original behavior) - 1D情况
    #     df_pred_true = pd.DataFrame({
    #         "true": list(trues_np),  
    #         "pred": list(preds_np)
    #     })
    
    # df_pred_true.to_csv(os.path.join(path_loss_save, 'TRUE_PRED_VALUE.csv'), index=True)
    # 创建保存目录
    os.makedirs(path_loss_save, exist_ok=True)
    
    # 先直接保存完整的原始数据为NumPy格式，避免内存问题
    # print(f"保存原始numpy数据，形状: {trues_np.shape}")
    # np.save(os.path.join(path_loss_save, f'{flag}_trues.npy'), trues_np)
    # np.save(os.path.join(path_loss_save, f'{flag}_preds.npy'), preds_np)
    
    if moe_weight is not None:
        pd.DataFrame(moe_weight.cpu().numpy()).to_csv(os.path.join(path_loss_save,'moe_weight.csv'), index=True)
    
    if labels is not None:
        # 处理有标签的情况，创建同时包含标签和损失的DataFrame
        if sensor_names is None:
            sensor_names = ['sensor3', 'sensor5', 'sensor6', 'sensor7', 'sensor10', 'sensor12', 'sensor14', 'sensor16']
        cols_data = {}
        
        # 确保labels和losses都是CPU张量
        labels_cpu = labels.cpu()
        losses_cpu = losses.cpu()
        
        # 检查维度是否匹配多传感器情况
        if labels_cpu.dim() > 1 and labels_cpu.shape[1] > 1:
            # 多传感器情况
            for i, sensor in enumerate(sensor_names):
                if i < labels_cpu.shape[1]:  # 确保索引在有效范围内
                    cols_data[f"{sensor}_label"] = labels_cpu[:, i].numpy()
                    cols_data[f"{sensor}_losses"] = losses_cpu[:, i].numpy()
        else:
            # 单传感器情况或已经聚合的情况
            cols_data[f"{sensor_names[0]}_label"] = labels_cpu.flatten().numpy()
            cols_data[f"{sensor_names[0]}_losses"] = losses_cpu.flatten().numpy()
            
        total_label_loss = pd.DataFrame(cols_data)
    else:
        # 只有损失没有标签的情况
        total_label_loss = losses.cpu()
        
        if total_label_loss.dim() > 1 and total_label_loss.shape[1] > 1:
            # 对于多传感器损失，为每个传感器创建列
            if sensor_names is None:
                sensor_names = ['sensor3', 'sensor5', 'sensor6', 'sensor7', 'sensor10', 'sensor12', 'sensor14', 'sensor16']
            columns_data = {}
            for i, sensor in enumerate(sensor_names):
                if i < total_label_loss.shape[1]:  # 确保索引在有效范围内
                    columns_data[f"{sensor}_losses"] = total_label_loss[:, i].numpy()
            total_label_loss = pd.DataFrame(columns_data)
        else:
            # 单传感器或已聚合损失的原始行为
            total_label_loss = total_label_loss.view(total_label_loss.shape[0], -1).numpy()
            total_label_loss = pd.DataFrame(total_label_loss)
    
    ts = pd.concat([timestamps, total_label_loss], axis=1)
    
    if flag == 'test' or flag == 'test_health':
        if flag == 'test':
            true_label = 1
        elif flag == 'test_health':
            true_label = 0
        
        # 添加true_label列，但不重命名已有列
        ts['true_label'] = true_label
        
        # 确保时间戳列名为'time'
        if 'time' not in ts.columns:
            # 假设第一列是时间戳
            cols = list(ts.columns)
            cols[0] = 'time'
            ts.columns = cols
            
        ts.to_csv(os.path.join(path_loss_save, 'label_loss.csv'), index=True)
        # 损伤识别 - 不再接收返回值
        damage_metric(flag, ts, path_loss_save)
    elif flag == 'vali':
        # 确保时间戳列名为'time'
        if 'time' not in ts.columns:
            # 假设第一列是时间戳
            cols = list(ts.columns)
            cols[0] = 'time'
            ts.columns = cols
        ts.to_csv(os.path.join(path_loss_save, 'label_loss.csv'), index=True)
    print("result saved!")

def plot_mae_distribution(csv_path, sensor_id, flag, plot_save_path):
    
    # 读入 CSV 文件。假设文件中第一行是表头，第一列为索引。
    df = pd.read_csv(csv_path, names=['time','pred_label', 'losses','true_label'], header=0, index_col=0)
    df['time'] = pd.to_datetime(df['time'], unit='ns')
    
    # 准备存储柱状图数据（均值、标准差、标签）
    mean_values = []
    std_values = []
    scenario_labels = []

    # 计算整个数据集中的最小和最大loss值，用于统一 x 轴范围
    global_min = df['losses'].min()
    global_max = df['losses'].max()
    # 如果全局最小值小于0（尽管你说没有负值），也保证下界为0
    # if global_min < 0:
    #     global_min = 0
    x = np.linspace(global_min, global_max, 300)  # 300个点，足够平滑

    damage_name = {'D1': 'Settlement depths of pier Koppigen,', 'D2': 'Spalling depths of concrete at soffit,', 'D3': 'Failure nums of anchor heads', 'D4': 'Rupture nums of tendons'}
    pier_damage_time = ['1998-08-10', '1998-08-12', '1998-08-17', '1998-08-18', '1998-08-19']
    pier_damage = ['D1_20mm', 'D1_40mm', 'D1_80mm', 'D1_95mm']
    spalling_damage_time = ['1998-08-25','1998-08-26', '1998-08-27']
    spalling_damage = ['Spalling of concrete at soffit, 12 m2', 'Spalling of concrete at soffit, 24 m2']
    anchor_damage_time = ['1998-09-02','1998-09-03','1998-09-04']
    anchor_damage = ['Failure of 2 anchor heads', 'Failure of 4 anchor heads']
    Rupture_damage_time = ['1998-09-07','1998-09-08','1998-09-09','1998-09-10']
    Rupture_damage = ['Rupture of 2 tendons', 'Rupture of 4 tendons', 'Rupture of 6 tendons']   

    if flag == 'D1':
        damage_time = pier_damage_time
        damage = pier_damage
    elif flag == 'D2':
        damage_time = spalling_damage_time
        damage = spalling_damage
    elif flag == 'D3':
        damage_time = anchor_damage_time
        damage = anchor_damage     
    elif flag == "D4":
        damage_time = Rupture_damage_time
        damage = Rupture_damage   
    else:
        raise ValueError(f"Unknown flag: {flag}")     
    # 定义各个损伤工况的时间边界和标签
    plt.figure(figsize=(8,6))
    plt.style.use('bmh')
    
    for i in range(len(damage_time) - 1):
        D_start = pd.to_datetime(damage_time[i])
        D_end   = pd.to_datetime(damage_time[i + 1])
        mask_D = (df['time'] >= D_start) & (df['time'] < D_end)
        df_D = df[mask_D]
        
        # 计算当前时间段内losses的均值和标准差
        mu = df_D['losses'].mean()
        sigma = df_D['losses'].std(ddof=1)
        
        # 存入列表，以便后续绘制柱状图
        mean_values.append(mu)
        std_values.append(sigma if not np.isnan(sigma) else 0.0)
        scenario_labels.append(damage[i])
        # 根据正态分布公式计算概率密度值
        y = norm.pdf(x, loc=mu, scale=sigma)
        plt.plot(x, y, label=damage[i])
    
    plt.xlabel('MAE (loss) distribution')
    plt.ylabel('Probability Density')
    plt.title(sensor_id + ' Loss Distribution for Different' + damage_name[flag])
    plt.legend()


    # 获取 csv 文件所在目录，并构造图片保存路径
    parent_dir = os.path.dirname(plot_save_path)
    os.makedirs(parent_dir, exist_ok=True)
    dir_one = os.path.join(parent_dir, 'mae_distribution')
    print(dir_one)
    os.makedirs(dir_one, exist_ok=True)
    file_name = f'loss_distribution_{flag}_sensor_{sensor_id}.png'
    save_path = os.path.join(dir_one, file_name)
    #print(save_path)
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
  
    # 6. 绘制柱状图（平均 MAE）
    plt.figure(figsize=(8,6))
    plt.bar(range(len(mean_values)), mean_values, yerr=std_values, capsize=5, alpha=0.7)
    plt.xticks(range(len(mean_values)), scenario_labels, rotation=30)
    plt.xlabel('Damage Scenarios')
    plt.ylabel('Average MAE')
    plt.title(f"{sensor_id} Average MAE Bar Chart ({damage_name[flag]})")
    
    # 7. 保存柱状图
    dir_two = os.path.join(parent_dir, 'bar_distribution')
    os.makedirs(dir_two, exist_ok=True)
    file_name = f'loss_bar_{flag}_sensor_{sensor_id}.png'
    save_path_bar = os.path.join(dir_two, file_name)
    plt.savefig(save_path_bar, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved distribution plot to: {save_path}")
    print(f"Saved bar chart to: {save_path_bar}")


# def box(loss, second_loss, flag_one, flag_two, path):
#     """
#     如果 second_loss 为 None，则只绘制一个直方图；
#     如果 second_loss 不为 None，则在同一图上叠加两个直方图。
#     """
#     import os
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch

#     os.makedirs(path, exist_ok=True)

#     # 如果输入为 torch.Tensor，则转换为 numpy 数组
#     if isinstance(loss, torch.Tensor):
#         loss = loss.detach().cpu().numpy()
#     if second_loss is not None and isinstance(second_loss, torch.Tensor):
#         second_loss = second_loss.detach().cpu().numpy()

#     # 假设 loss 的形状为 (num_samples, 1)
#     if loss.shape[1] == 1:
#         plt.figure(figsize=(8, 6))
#         plt.style.use('bmh')
#         # 合并数据，便于确定 x 轴范围
#         if second_loss is None:
#             combined_loss = loss.flatten()
#         else:
#             combined_loss = np.concatenate([loss.flatten(), second_loss.flatten()])
#         # 计算 1% 和 99% 分位数，避免极端值干扰
#         x_min = np.percentile(combined_loss, 1)
#         x_max = np.percentile(combined_loss, 99)
        
#         if second_loss is None:
#             plt.hist(loss, bins=2000, edgecolor='black')
#             plt.xlabel('Loss value')
#             plt.ylabel('nums')
#             plt.title(flag_one + '  Loss Histogram')
#             plt.xlim(x_min, x_max)
#         else:
#             plt.hist(loss, bins=2000, alpha=0.5, edgecolor='purple', label='Loss_' + flag_one)
#             plt.hist(second_loss, bins=2000, alpha=0.5, edgecolor='orange', label='Loss_' + flag_two)
#             plt.xlabel('Loss value')
#             plt.ylabel('nums')
#             plt.title(flag_one + ' and ' + flag_two + '  Loss Histogram Comparison')
#             plt.legend()
#             plt.xlim(x_min, x_max)
#         save_name = flag_one + '_' + flag_two + 'loss对比直方图.png'
#         plt.savefig(os.path.join(path, save_name), dpi=300, bbox_inches='tight')
#         plt.close()

def box(loss, second_loss, third_loss=None, flag_one='', flag_two='', flag_three='', path='', lower_percentile=0, upper_percentile=80):
    """
    如果 third_loss 为 None，则根据 second_loss 是否为 None 来绘制一个或两个直方图；
    如果 third_loss 不为 None，则绘制三个直方图。
    支持通过 lower_percentile 和 upper_percentile 调整 xlim 范围，默认 1% 和 99% 分位数。
    """


    os.makedirs(path, exist_ok=True)

    if isinstance(loss, torch.Tensor):
        loss = loss.detach().cpu().numpy()
    if second_loss is not None and isinstance(second_loss, torch.Tensor):
        second_loss = second_loss.detach().cpu().numpy()
    if third_loss is not None and isinstance(third_loss, torch.Tensor):
        third_loss = third_loss.detach().cpu().numpy()


    plt.figure(figsize=(8, 6))
    plt.style.use('bmh')

    def remove_extremes(data, lower_percentile, upper_percentile):
        x_min = np.percentile(data, lower_percentile)
        x_max = np.percentile(data, upper_percentile)
        return data[(data >= x_min) & (data <= x_max)]
    
    loss = np.asarray(loss)
    second_loss = np.asarray(second_loss)
    third_loss = np.asarray(third_loss)
    loss_clean = remove_extremes(loss.flatten(), lower_percentile, upper_percentile)
    if second_loss is not None:
        second_loss_clean = remove_extremes(second_loss.flatten(), lower_percentile, upper_percentile)
    if third_loss is not None:
        third_loss_clean = remove_extremes(third_loss.flatten(), lower_percentile, upper_percentile)
    
    combined_loss = loss_clean
    if second_loss is not None:
        combined_loss = np.concatenate([combined_loss, second_loss_clean])
    if third_loss is not None:
        combined_loss = np.concatenate([combined_loss, third_loss_clean])
    x_min = np.percentile(combined_loss, lower_percentile)
    x_max = np.percentile(combined_loss, upper_percentile)
    plt.hist(loss_clean, bins=200, alpha=0.6, color='purple', label='Loss_' + flag_one)
    plt.hist(second_loss_clean, bins=200, alpha=0.6, color='orange', label='Loss_' + flag_two)
    plt.hist(third_loss_clean, bins=200, alpha=0.3, color='green', label='Loss_' + flag_three)
    plt.xlabel('Loss value')
    plt.ylabel('nums')
    plt.title(flag_one + ', ' + flag_two + ' and ' + flag_three + '  Loss Histogram Comparison')
    plt.legend()
    save_name = flag_one + '_' + flag_two + '_' + flag_three + 'loss对比直方图.png'
    
    plt.xlim(x_min, x_max)  # 根据合并数据的百分位数范围设置 xlim
    plt.savefig(os.path.join(path, save_name), dpi=300, bbox_inches='tight')
    plt.close()

# def plot_true_pred(csv_path, save_path, flag):
#     filename = os.path.join(save_path, 'true_value', flag + 'true_pred_comparation.png')
#     data = pd.read_csv(csv_path)
#     true_value = data['true']
#     reconstructed_value = data['pred']
#     plt.figure(figsize=(10, 6))
#     plt.style.use('bmh')
#     plt.plot(true_value, label='True Value', color='blue', linestyle='-', marker='o')
#     plt.plot(reconstructed_value, label='Reconstructed Value', color='purple', linestyle='--', marker='x')
#     plt.title(flag + 'True Value vs Reconstructed Value')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(filename) 
#     plt.close

def save_segment(idx, segment_name, true_value, pred_value,window_size,mse_array,flag,save_path):
    true_seg = true_value[idx:idx+window_size]
    pred_seg_raw = pred_value[idx:idx+window_size]
    bias = np.mean(pred_seg_raw) - np.mean(true_seg)
    pred_seg = pred_seg_raw - bias
    plt.figure(figsize=(10, 6))
    plt.style.use('bmh')
    plt.plot(true_seg, label='True Value', color='#8b2e2e', linestyle='-',marker=None)
    plt.plot(pred_seg, label='Reconstructed Value', color='#2e8b92', linestyle='--',marker=None)
    plt.title(f"{flag} {segment_name} (MSE: {mse_array[idx]:.4f})")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(save_path, 'plot_comparation', f'{flag}_{segment_name}.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"{segment_name} 图已保存至: {fig_path}")

    segment_data = pd.DataFrame({
        'true': true_seg,
        'pred': pred_seg
    })
    csv_segment_path = os.path.join(save_path, 'plot_comparation', f'{flag}_{segment_name}.csv')
    segment_data.to_csv(csv_segment_path, index=False)
    print(f"{segment_name} 数据已保存至: {csv_segment_path}")
def plot_true_pred(csv_path, save_path, flag):
    os.makedirs(os.path.join(save_path, 'plot_comparation'), exist_ok=True)
    window_size=1000
    data = pd.read_csv(csv_path)
    true_value = data['true'].apply(lambda x: float(x.strip('[]'))).values
    pred_value = data['pred'].apply(lambda x: float(x.strip('[]'))).values
    num_windows = len(true_value) - window_size + 1
    mse_list = []
    # bias = np.mean(pred_value) - np.mean(true_value)
    # pred_value_aligned = pred_value - bias
    for i in range(num_windows):
        true_win = true_value[i:i+window_size]
        pred_win = pred_value[i:i+window_size]
        mse = np.mean((true_win - pred_win) ** 2)
        mse_list.append(mse)
    mse_array = np.array(mse_list)
    best_idx = np.argmin(mse_array)
    worst_idx = np.argmax(mse_array)
    save_segment(best_idx, 'most_similar_segment', true_value, pred_value,window_size,mse_array,flag,save_path)
    save_segment(worst_idx, 'most_dissimilar_segment', true_value, pred_value,window_size,mse_array,flag,save_path)


def plot_loss(train_loss=None, vali_loss=None, test_health_vali_loss=None, test_vali_loss=None, folder='', patience='', loss_type=''):
    """
    绘制并保存 loss 曲线，所有的 loss 参数均为可选输入。
    对于每个epoch中包含多个传感器损失的情况，计算平均损失后再绘图。

    参数:
        train_loss: 训练 loss(数组或列表)，可选
        vali_loss: 验证 loss(数组或列表)，可选
        test_health_vali_loss: 测试健康集 loss(数组或列表)，可选
        test_vali_loss: 测试 loss(数组或列表)，可选
        folder: 保存图像的基础路径
        patience: 用于文件名的 patience 参数
        loss_type: 用于文件名的损失类型字符串
    """
    import numpy as np

    # 如果有测试相关的 loss，则保存路径设为子文件夹 "tvt_loss_comparation"
    if test_health_vali_loss is not None or test_vali_loss is not None:
        plot_path = os.path.join(folder, 'test_loss_comparation')
    else:
        plot_path = os.path.join(folder, 'train_vali_comparation')
    os.makedirs(plot_path, exist_ok=True)

    plt.figure()
    plt.style.use('bmh')
    title_parts = []
    
    # 处理训练损失
    if train_loss is not None:
        # 检查是否是多维数组，如果是则计算平均值
        avg_train_loss = []
        for epoch_loss in train_loss:
            if hasattr(epoch_loss, '__len__') and not isinstance(epoch_loss, (str, bytes)):
                # 如果epoch_loss是可迭代对象（如列表、数组等），计算平均值
                avg_train_loss.append(np.mean(epoch_loss))
            else:
                # 如果是单个数值，直接添加
                avg_train_loss.append(epoch_loss)
        
        plt.plot(avg_train_loss, label='Train Loss (Avg)', color='orange', linestyle='--')
        title_parts.append('Train Loss (Avg)')
    
    # 处理验证损失
    if vali_loss is not None:
        # 检查是否是多维数组，如果是则计算平均值
        avg_vali_loss = []
        for epoch_loss in vali_loss:
            if hasattr(epoch_loss, '__len__') and not isinstance(epoch_loss, (str, bytes)):
                # 如果epoch_loss是可迭代对象，计算平均值
                avg_vali_loss.append(np.mean(epoch_loss))
            else:
                # 如果是单个数值，直接添加
                avg_vali_loss.append(epoch_loss)
        
        plt.plot(avg_vali_loss, label='Validation Loss (Avg)', color='purple', linestyle='-.')
        title_parts.append('Validation Loss (Avg)')
    
    # 处理测试健康损失
    if test_health_vali_loss is not None:
        # 检查是否是多维数组，如果是则计算平均值
        avg_test_health_loss = []
        for epoch_loss in test_health_vali_loss:
            if hasattr(epoch_loss, '__len__') and not isinstance(epoch_loss, (str, bytes)):
                # 如果epoch_loss是可迭代对象，计算平均值
                avg_test_health_loss.append(np.mean(epoch_loss))
            else:
                # 如果是单个数值，直接添加
                avg_test_health_loss.append(epoch_loss)
        
        plt.plot(avg_test_health_loss, label='Test Health Loss (Avg)', color='pink', linestyle='-')
        title_parts.append('Test Health Loss (Avg)')
    
    # 处理测试损失
    if test_vali_loss is not None:
        # 检查是否是多维数组，如果是则计算平均值
        avg_test_loss = []
        for epoch_loss in test_vali_loss:
            if hasattr(epoch_loss, '__len__') and not isinstance(epoch_loss, (str, bytes)):
                # 如果epoch_loss是可迭代对象，计算平均值
                avg_test_loss.append(np.mean(epoch_loss))
            else:
                # 如果是单个数值，直接添加
                avg_test_loss.append(epoch_loss)
        
        plt.plot(avg_test_loss, label='Test Loss (Avg)', color='purple', linestyle=':')
        title_parts.append('Test Loss (Avg)')
    
    # 如果没有任何 loss 数据，则直接返回
    if not title_parts:
        # print("未提供任何 loss 数据，无法绘图。")
        return

    title = ', '.join(title_parts)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    # 保存图像
    filename = f"epoch_loss_{loss_type}_{patience}.png"
    plt.savefig(os.path.join(plot_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def percentile_cal(loss, per_num):  
    if isinstance(loss, pd.Series):
        loss_np = loss.to_numpy()
        loss_mean = np.mean(loss_np, axis=0)
        percentile = np.percentile(loss_np, per_num, axis=0)
    else:
        loss_list = [loss_part.cpu().detach() for loss_part in loss]
        loss_tensor = torch.stack(loss_list, dim=0)
        loss_mean = loss_tensor.mean(dim=0).numpy()
        percentile = np.percentile(loss_tensor.numpy(), per_num, axis=0)
    return loss_mean, percentile

def plot_show(clean_data, noisy_data):
    plt.figure(figsize=(12, 6))
    plt.plot(clean_data[0].cpu(), label='Clean', alpha=0.6)
    plt.plot(noisy_data[0].cpu(), label='Noisy', alpha=0.8)
    plt.title("多类型传感器故障注入示例")
    plt.legend()
    plt.show()
    plt.close()
    
def loss_save(loss:torch.Tensor, folder:str, flag:str):
    np = loss.cpu().detach().numpy()
    df = pd.DataFrame(np)
    path = os.path.join(folder, flag, 'loss.csv')
    os.makedirs(path, exist_ok=True)
    df.to_csv(path, index=False)


# class Bridge_loss(nn.Module):

#     def __init__(self,error_type: str = 'mse'):
#         super(Bridge_loss, self).__init__()
#         if error_type not in ['mse', 'mae', 'rmse']:
#             raise ValueError(f"Invalid error_type: {error_type}")
#         self.error_type = error_type

#     def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#         # 假设 pred 和 true 的形状均为 (8193, 2048, 8)
#         # 如果使用 torch 操作，下面代码也可以直接用 torch 来实现
        # if self.error_type in ['mse', 'rmse']:
        #     calculation = (pred - true) ** 2
        # elif self.error_type == 'mae':
        #     calculation = torch.abs(pred - true)
        
        # # 沿第二维（axis=1）累加误差-> shape: (batch_size, feature_nums)
        # error_sum = torch.sum(calculation, dim=1, keepdim=False)
        # # 沿第一维（axis=0）求均值 -> shape: (feature_nums)
        # error_mean = torch.mean(error_sum, dim=0, keepdim=False)
        
#         # 如果是 rmse，需要取平方根
#         if self.error_type == 'rmse':
#             final_error = torch.sqrt(error_mean)
#         else:
#             final_error = error_mean
        
#         return final_error

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bridge_Loss(nn.Module):
    def __init__(self, loss_mode='mae', alpha=1, beta=0.5, gamma=0.3):
        super(Bridge_Loss, self).__init__()
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.beta = beta

        if loss_mode not in ['mse', 'mae', 'rmse', 'freq_mse', 'combined']:
            raise ValueError(f"Invalid loss_mode: {loss_mode}")

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: shape = [B, L, D], e.g. [batch_size, seq_len, sensor_features]
        return: shape = [1, 1, D]
        """
        if self.loss_mode == 'mse':
            return self._reduce((pred - true) ** 2)

        elif self.loss_mode == 'mae':
            return self._reduce(torch.abs(pred - true))

        elif self.loss_mode == 'rmse':
            return torch.sqrt(self._reduce((pred - true) ** 2))

        elif self.loss_mode == 'freq_mse':
            return self.freq_loss(pred, true)

        elif self.loss_mode == 'combined':
            # 使用MAE作为默认空间域损失，与频域损失结合
            return (
                self.alpha * self._reduce(torch.abs(pred - true)) +
                self.beta * self.freq_loss(pred, true)
            )

    def _reduce(self, error_tensor: torch.Tensor):

        error_sum = torch.sum(error_tensor, dim=1)        # -> [B, D]
        error_mean = torch.mean(error_sum, dim=0)         # -> [D]
        return error_mean.view(-1)                  # -> [D]

    def freq_loss(self, pred, true):
        fft_pred = torch.fft.rfft(pred, dim=1)
        fft_true = torch.fft.rfft(true, dim=1)
        # 改用MAE计算频域损失
        error = torch.abs(torch.abs(fft_pred) - torch.abs(fft_true))
        error_sum = torch.sum(error, dim=1)       # [B, D]
        error_mean = torch.mean(error_sum, dim=0) # [D]
        return error_mean.view(-1)

class Bridge_Loss_Vali(nn.Module):
    def __init__(self, error_type: str = 'mse'):
        super(Bridge_Loss_Vali, self).__init__()
        if error_type not in ['mse', 'mae', 'rmse', 'orsr', 'freq','corr']:
            raise ValueError(f"Invalid error_type: {error_type}")
        self.error_type = error_type

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: shape = [B, L, D], e.g. [8193, 2048, 8]
        Returns: shape = [B, D] or [B, 3, D] (for 'combine')
        """
        if self.error_type in ['mse', 'rmse', 'mae']:
            if self.error_type == 'mse':
                calculation = (pred - true) ** 2
            elif self.error_type == 'mae':
                calculation = torch.abs(pred - true)
            elif self.error_type == 'rmse':
                calculation = (pred - true) ** 2
            error_sum = torch.sum(calculation, dim=1)  # [B, D]
            if self.error_type == 'rmse':
                final_error = torch.sqrt(error_sum)
            else:
                final_error = error_sum
            return final_error  # [B, D]

        elif self.error_type == 'orsr':
            x2_sum = torch.sum(true ** 2, dim=1) + 1e-8 
            y2_sum = torch.sum(pred ** 2, dim=1) + 1e-8
            orsr = torch.abs(10 * torch.log10(y2_sum/ x2_sum))
            return orsr  # [B, D]

        elif self.error_type == 'freq':
            pred_freq = torch.fft.rfft(pred, dim=1)
            true_freq = torch.fft.rfft(true, dim=1)
            freq_diff = (torch.abs(pred_freq - true_freq)) ** 2
            freq_error = torch.sum(freq_diff, dim=1)
            return freq_error  # [B, D]
        elif self.error_type == 'corr':
            corr = torch.corrcoef(pred, true)
            return 1 - torch.abs(corr)  # [B, D]

        else:
            raise ValueError("Unsupported error type")
        
class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim):
        super(MoE, self).__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList([self._create_expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.router = self._create_router(input_dim, hidden_dim)
        self.expert_prototypes = nn.Parameter(torch.randn(num_experts, input_dim))

    def _create_expert(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def _create_router(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def compute_similarity(self, x, prototypes):
        x_norm = F.normalize(x, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        similarity = torch.matmul(x_norm, proto_norm.T)  # [batch_size, num_experts]
        return similarity
    
    def forward(self, x):
        router_output = self.router(x)  # [batch_size, input_dim]
        routing_weights = self.compute_similarity(router_output, self.expert_prototypes)  # [batch_size, num_experts]
        
        routing_weights_softmax = F.softmax(routing_weights, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0).transpose(0, 1)  # [batch_size, num_experts, input_dim]
        weighted_output = torch.matmul(routing_weights_softmax.unsqueeze(1), expert_outputs).squeeze(1)  # [batch_size, input_dim]
        return weighted_output, routing_weights_softmax


def heatmap_plot(moe_weight_list, flag_list, save_path):

    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)
    dir_moe = os.path.join(parent_dir, 'moe_weight_heatmap')
    os.makedirs(dir_moe, exist_ok=True)
 
    for i in range(len(moe_weight_list)):
        file_name = f'moe_weight_{flag_list[i]}.png'
        save_path = os.path.join(dir_moe, file_name)

        plt.figure(figsize=(8, 6)) 
        plt.style.use('bmh')

        sns.heatmap(moe_weight_list[i].to_numpy(), annot=False, cmap='coolwarm', fmt='.2f', linewidth=0.5)
        plt.title(f'heatmep of {flag_list[i]} data')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def visualize_attention(attn_file_path, save_dir=None):
    """
    可视化保存的注意力权重文件
    
    参数:
        attn_file_path: 注意力权重文件路径(.pt格式)
        save_dir: 可视化结果保存目录，如果为None则使用attn_file所在目录
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    # 加载注意力文件
    attn_data = torch.load(attn_file_path)
    
    # 如果未指定保存目录，使用输入文件的目录
    if save_dir is None:
        save_dir = os.path.dirname(attn_file_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取文件名(不含扩展名)作为基础保存名
    base_filename = os.path.splitext(os.path.basename(attn_file_path))[0]
    
    # 获取数据集类型和温度范围信息
    dataset_type = attn_data.get('dataset_type', 'unknown')
    temp_range = attn_data.get('temp_range', (0, 0))
    
    # 打印文件基本信息
    print(f"分析注意力文件: {attn_file_path}")
    print(f"数据集类型: {dataset_type}")
    print(f"温度范围: {temp_range}")
    print(f"样本数量: {attn_data.get('count', 0)}")
    
    # 查找文件中的注意力权重
    attn_keys = [k for k in attn_data.keys() if isinstance(attn_data[k], torch.Tensor) and 
                'attention' in k.lower() or k in ['attention', 'attn']]
    
    if not attn_keys:
        print("警告: 文件中未找到注意力权重数据")
        return
    
    # 为每个注意力权重创建可视化
    for attn_key in attn_keys:
        attn_weights = attn_data[attn_key]
        
        # 如果是3D+的张量，取第一个样本
        if attn_weights.dim() > 2:
            # 对于更复杂的多头注意力情况
            if attn_weights.dim() >= 4:  # [batch, heads, seq_len, seq_len]
                # 取第一个样本的平均头注意力
                attn_weights = attn_weights[0].mean(dim=0)
            else:  # [batch, seq_len, seq_len]
                attn_weights = attn_weights[0]
                
        # 确保是2D张量
        if attn_weights.dim() == 1:
            # 如果是1D，转为方形矩阵
            seq_len = int(np.sqrt(len(attn_weights)))
            attn_weights = attn_weights.reshape(seq_len, seq_len)
        
        # 转为numpy进行可视化
        attn_np = attn_weights.detach().cpu().numpy()
        
        # 创建自定义的ColorMap
        colors = [(0.00, '#F2F0E5'), (0.50, '#4E79A7'), (0.75, '#E15759'), (1.00, '#B07AA1')]
        custom_cmap = LinearSegmentedColormap.from_list('custom_attention', colors)
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            attn_np, 
            cmap=custom_cmap,
            square=True,
            vmin=0,
            vmax=None,  # 自动适应最大值
            center=None,
            annot=False,
            fmt=".2f"
        )
        
        # 设置标题和轴标签
        plt.title(f"{dataset_type} - {attn_key} 热力图 (温度: {temp_range[0]}-{temp_range[1]}°C)")
        
        # 根据注意力矩阵大小调整轴标签
        seq_len = attn_np.shape[0]
        if seq_len <= 20:  # 只在序列长度合理时显示所有刻度
            plt.xticks(np.arange(seq_len) + 0.5, np.arange(seq_len))
            plt.yticks(np.arange(seq_len) + 0.5, np.arange(seq_len))
            ax.set_xlabel('序列位置')
            ax.set_ylabel('序列位置')
        else:  # 对于长序列，只显示部分刻度
            tick_step = max(1, seq_len // 10)
            plt.xticks(np.arange(0, seq_len, tick_step) + 0.5, np.arange(0, seq_len, tick_step))
            plt.yticks(np.arange(0, seq_len, tick_step) + 0.5, np.arange(0, seq_len, tick_step))
            ax.set_xlabel('序列位置 (下采样)')
            ax.set_ylabel('序列位置 (下采样)')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f"{base_filename}_{attn_key}_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"保存热力图至: {save_path}")
        
        # 如果注意力权重不是太大，创建一个3D表面图
        if seq_len <= 50:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 创建网格
            x = np.arange(attn_np.shape[1])
            y = np.arange(attn_np.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # 绘制3D表面
            surf = ax.plot_surface(X, Y, attn_np, cmap=custom_cmap, 
                                  linewidth=0, antialiased=True, alpha=0.8)
            
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # 设置标题和轴标签
            ax.set_title(f"{dataset_type} - {attn_key} 3D表面图 (温度: {temp_range[0]}-{temp_range[1]}°C)")
            ax.set_xlabel('目标位置')
            ax.set_ylabel('查询位置')
            ax.set_zlabel('注意力权重')
            
            if seq_len > 10:
                tick_step = max(1, seq_len // 10)
                ax.set_xticks(np.arange(0, seq_len, tick_step))
                ax.set_yticks(np.arange(0, seq_len, tick_step))
            
            # 保存3D图
            save_path_3d = os.path.join(save_dir, f"{base_filename}_{attn_key}_3d.png")
            plt.savefig(save_path_3d, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"保存3D表面图至: {save_path_3d}")

    print(f"所有注意力可视化已保存至: {save_dir}")
    
    return save_dir

def batch_visualize_attention(attention_dir, pattern='*.pt', save_subdir='visualized'):
    """
    批量可视化目录中的所有注意力权重文件
    
    参数:
        attention_dir: 包含注意力权重文件的目录
        pattern: 文件匹配模式，默认为'*.pt'
        save_subdir: 保存可视化结果的子目录名
    """
    import os
    import glob
    
    # 确保目录存在
    os.makedirs(attention_dir, exist_ok=True)
    
    # 创建保存目录
    save_dir = os.path.join(attention_dir, save_subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 查找所有匹配的文件
    search_path = os.path.join(attention_dir, pattern)
    pt_files = glob.glob(search_path)
    
    if not pt_files:
        print(f"警告: 在 {attention_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"找到 {len(pt_files)} 个注意力文件，开始批量可视化...")
    
    # 处理每个文件
    for pt_file in pt_files:
        try:
            visualize_attention(pt_file, save_dir)
            print(f"已处理: {os.path.basename(pt_file)}")
        except Exception as e:
            print(f"处理 {os.path.basename(pt_file)} 时出错: {e}")
    
    print(f"批量可视化完成! 结果保存在: {save_dir}")
    return save_dir