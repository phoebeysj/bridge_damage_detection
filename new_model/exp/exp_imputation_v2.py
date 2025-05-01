from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, inject_sensor_fault, diversity_loss, result_save, plot_loss, percentile_cal, plot_show, loss_save, Bridge_Loss, Bridge_Loss_Vali, MoE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial import ConvexHull
import gc
warnings.filterwarnings('ignore')




class Exp_Imputation_bridge(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation_bridge, self).__init__(args)
        self.fault_config = {
            'normal': 0.7,
            'missing': 0.07,
            'minor': 0.05,
            'square': 0.05,
            'trend': 0.05,
            'spike': 0.05,
            'outlier': 0.03,
        }
        self.div = 0.2
        self.loss_mode = args.loss_mode

        # 打印GPU信息和内存使用情况
        if args.use_gpu:
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            
        # 如果使用多GPU，确保批量大小合适
        if args.use_multi_gpu and args.use_gpu and hasattr(args, 'device_ids'):
            num_gpus = len(args.device_ids) if isinstance(args.device_ids, list) else len(args.device_ids.split(','))
            if num_gpus > 1:
                print(f"Using {num_gpus} GPUs")
                # 确保batch_size是GPU数量的整数倍
                if hasattr(args, 'batch_size') and args.batch_size % num_gpus != 0:
                    original_batch_size = args.batch_size
                    args.batch_size = (args.batch_size // num_gpus) * num_gpus
                    print(f"Adjusted batch_size from {original_batch_size} to {args.batch_size} to fit {num_gpus} GPUs")

        print(f"可见GPU数量: {torch.cuda.device_count()}")

    def align_and_stack_wavelet_coeffs(self, coeff_dict):
        """
        对齐并堆叠小波分量张量，支持动态尺度数
        Args:
            coeff_dict: dict, 例如 {'cA{n}': [B, T1, N], 'cD{n}': [B, T2, N], 'cD{n-1}': [B, T3, N], ...}
                    其中n为最大尺度数
        Returns:
            tensor: [B, max_T, N, n_scales+1]  # n_scales+1为分量数（1个近似系数 + n_scales个细节系数）
        """
        # 1. 确定尺度数
        scales = []
        max_scale = 0
        for k in coeff_dict.keys():
            if k.startswith('cA'):
                max_scale = int(k[2:])  # 从'cA{n}'中提取n
            scales.append(k)
        
        # 2. 按顺序排列系数名称：[cA{n}, cD{n}, cD{n-1}, ..., cD1]
        ordered_scales = [f'cA{max_scale}']  # 首先是最高尺度的近似系数
        for scale in range(max_scale, 0, -1):  # 然后是从高到低的细节系数
            ordered_scales.append(f'cD{scale}')
        
        # 3. 找到最大长度
        lengths = [v.shape[1] for v in coeff_dict.values()]
        max_len = max(lengths)
        batch_size, num_sensors = next(iter(coeff_dict.values())).shape[0], next(iter(coeff_dict.values())).shape[2]
        
        # 4. 处理每个尺度的系数
        out_list = []
        for scale_name in ordered_scales:
            v = coeff_dict[scale_name]  # [B, T, N]
            repeat_factor = max_len // v.shape[1]
            remainder = max_len % v.shape[1]
            # 先repeat
            v_rep = v.repeat_interleave(repeat_factor, dim=1)
            # 补齐remainder
            if remainder > 0:
                v_rep = torch.cat([v_rep, v[:, :remainder, :]], dim=1)
            out_list.append(v_rep.unsqueeze(1))  # [B, max_T, N, 1]
        
        # 5. 拼接所有尺度的系数
        out = torch.cat(out_list, dim=1)  # [B, max_T, N, n_scales+1]
        return out 

    def _process_batch_and_collect_features(self, batch_x_mark, batch_env, batch_y_mark, criterion,
                                          anomaly_threshold=None, return_details=False, **wavelet_coeffs):
        """处理批次数据并收集特征
        Args:
            batch_x_mark: 时间特征标记,形状为 [B, T, N_time_mark]
            batch_env: 环境数据,形状为 [B, T, N_env]
            batch_y_mark: 时间戳,形状为 [B, T, 1]  
            criterion: 损失函数
            anomaly_threshold: 异常检测阈值
            return_details: 是否返回详细信息
            **wavelet_coeffs: 动态接收不同尺度的小波系数
                格式为: {
                    'cA{n_scales}': 最后一层近似系数,
                    'cD{n_scales}': 第n_scales层细节系数,
                    'cD{n_scales-1}': 第n_scales-1层细节系数,
                    ...
                    'cD1': 第1层细节系数
                }
        """
        # 为每个尺度系数创建掩码
        masks = {
            name: torch.rand(coeff.shape).to(self.device)
            for name, coeff in wavelet_coeffs.items()
        }
        
        # 应用掩码阈值
        for mask in masks.values():
            mask[mask <= self.args.mask_rate] = 0  # masked
            mask[mask > self.args.mask_rate] = 1  # remained
        
        # 构建输入字典
        # batch_x = wavelet_coeffs
        batch_x = self.align_and_stack_wavelet_coeffs(wavelet_coeffs)
        B, n_scales, T, N = batch_x.shape
        # 获取模型输出
        outputs, hidden_states, attention_weights = self.model(
            batch_x, batch_x_mark, None, None, masks, batch_env
        )
        
        # # 计算每个尺度的损失
        # losses = {}
        # total_loss = 0
        # for name, coeff in batch_x.items():
        #     if name in outputs:
        #         loss = criterion(outputs[name].squeeze(1), coeff)
        #         losses[name] = loss
        #         total_loss += loss

        total_loss = criterion(outputs.reshape(B, n_scales * T, N), batch_x.reshape(B, n_scales * T, N))
        # 如果只需要返回损失相关信息（用于训练和验证）
        if not return_details:
            return {
                'loss': total_loss,
                # 'individual_losses': losses
            }
        
        # 如果需要返回完整信息（用于测试和分析）
        # 计算异常标签（如果需要）
        sample_labels = None
        if anomaly_threshold is not None:
            sample_labels = total_loss > torch.Tensor(anomaly_threshold).to(self.device)
        
        # 准备返回数据
        pred = {name: output.detach().cpu() for name, output in outputs.items()}
        true = {name: coeff.detach().cpu() for name, coeff in batch_x.items()}
        masks_cpu = {name: mask.detach().cpu() for name, mask in masks.items()}
        
        # 处理时间戳
        time_stamp = batch_y_mark
        data_stamp = pd.to_datetime(time_stamp[:,0].cpu().numpy(), unit='ns')
        
        # 收集环境数据
        temps = batch_env[:, :, 3].cpu().numpy()  # 气温AT (index 3)
        humidity = None
        if hasattr(batch_env, 'shape') and batch_env.shape[1] > 2:
            humidity = batch_env[:,:, 2].cpu().numpy()  # 湿度 H (索引2)
        
        return {
            'pred': pred,
            'true': true,
            'mask': masks_cpu,
            'loss': total_loss.detach(),
            # 'individual_losses': losses,
            'sample_labels': sample_labels,
            'data_stamp': data_stamp,
            'hidden_states': hidden_states,
            'attention_weights': attention_weights,
            'temps': temps,
            'humidity': humidity
        }
    
    def _process_temp_binning(self, batch_result, temps, temp_bins, temp_binned_data, completed_bins, samples_per_bin):
        """处理温度分箱数据收集"""
        print(f"开始处理温度分箱数据")
        if self.args.use_gpu:
            print(f"GPU内存状态: 已分配 {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        batch_x = batch_result['true']  # 字典格式 {'cA3': [B,T,N], 'cD3': [B,T,N]...}
        attention_weights = batch_result['attention_weights']
        hidden_states = batch_result['hidden_states']
        
        # 获取批次大小
        B = next(iter(batch_x.values())).shape[0]
        
        # 计算每个样本的平均温度和湿度
        mean_temps = np.mean(temps, axis=1)  # [B]
        humidity = batch_result.get('humidity', None)
        mean_humidity = np.mean(humidity, axis=1) if humidity is not None else None  # [B]
        
        # 初始化温度分箱数据结构
        if not temp_binned_data:
            temp_binned_data = {i: {
                'cA3': [], 'cD3': [], 'cD2': [], 'cD1': [],  # 小波数据
                'temp': [], 
                'humidity': [], 
                'spatial_attention': [],  # 每个箱的空间注意力
                'temporal_attention': [], # 每个箱的时间注意力
                'env_attention': [],  # 环境注意力
                'features': []  # 隐藏特征
            } for i in range(len(temp_bins))}
        
        # 为每个样本找到对应的温度箱
        for j in range(B):
            mean_temp = mean_temps[j]
            bin_idx = None
            
            # 找到对应的温度箱
            for k, (bin_min, bin_max) in enumerate(temp_bins):
                if bin_min <= mean_temp < bin_max:
                    bin_idx = k
                    break
            
            # 如果找到有效的温度箱且箱未满，收集样本
            if bin_idx is not None and bin_idx not in completed_bins:
                bin_data = temp_binned_data[bin_idx]
                
                # 检查所有尺度是否都未满
                scales = ['cA3', 'cD3', 'cD2', 'cD1']
                all_scales_not_full = all(
                    len(bin_data[scale]) < samples_per_bin 
                    for scale in scales
                )
                
                if all_scales_not_full:
                    # 存储小波数据
                    for scale in scales:
                        bin_data[scale].append(
                            batch_x[scale][j:j+1].detach().cpu()
                        )
                    
                    # 存储温度和湿度
                    bin_data['temp'].append(mean_temp)
                    if mean_humidity is not None:
                        bin_data['humidity'].append(mean_humidity[j])
                    
                    # 处理注意力权重
                    if attention_weights is not None and isinstance(attention_weights, dict):
                        # 空间注意力
                        spatial_attn = [
                            attn[j:j+1,:,:].detach().cpu() 
                            for attn in attention_weights['spatial']
                        ]
                        bin_data['spatial_attention'].append(spatial_attn)
                        
                        # 时间注意力
                        temporal_attn = [
                            attn[j:j+1,:,:].detach().cpu() 
                            for attn in attention_weights['temporal']
                        ]
                        bin_data['temporal_attention'].append(temporal_attn)
                        
                        # # 环境注意力 - 修改这部分
                        # env_attn = []
                        # for attn in attention_weights['env_cross_attention']:
                        #     # 确保attn是张量而不是列表
                        #     if isinstance(attn, torch.Tensor):
                        #         env_attn.append(attn[j:j+1].detach().cpu())
                        #     else:
                        #         # 如果是列表，先转换为张量
                        #         attn_tensor = torch.stack(attn)
                        #         env_attn.append(attn_tensor[j:j+1].detach().cpu())
                        # bin_data['env_attention'].append(env_attn)
                    
                    # 处理隐藏特征
                    if hidden_states is not None:
                        if isinstance(hidden_states, dict) and 'fused' in hidden_states:
                            bin_data['features'].append(
                                hidden_states['fused'][j:j+1,:,:].detach().cpu()
                            )
                
                    # 检查是否所有尺度都已满
                    if all(
                        len(bin_data[scale]) >= samples_per_bin 
                        for scale in batch_x.keys()
                    ):
                        completed_bins.add(bin_idx)
        
        # 每个批次结束后强制清理
        import gc
        gc.collect()
        if self.args.use_gpu:
            torch.cuda.empty_cache()
        
        print(f"温度分箱处理完成，当前内存状态: 已分配 {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    def _save_global_features(self, features_list, temps_list, hums_list, labels_list, 
                             dataset_type, path_loss_save):
        """合并并保存全局特征数据"""
        if not features_list:
            return None
            
        try:
            # 首先将特征列表中的所有张量拼接在一起
            if isinstance(features_list[0], torch.Tensor):
                features = torch.cat(features_list, dim=0)
            else:
                # 如果特征列表中包含的不是直接的张量而是列表，先将内部列表中的张量拼接
                all_features = []
                for feature_batch in features_list:
                    if isinstance(feature_batch, list):
                        all_features.extend(feature_batch)
                    else:
                        all_features.append(feature_batch)
                features = torch.cat(all_features, dim=0)
            
            temps = np.array(temps_list)
            hums = np.array(hums_list) if hums_list else np.zeros_like(temps)
            labels = np.array(labels_list)
            
            features_save_path = os.path.join(path_loss_save, 'latent_features')
            os.makedirs(features_save_path, exist_ok=True)
            
            global_save_data = {
                'features': features,
                'temperature': temps,
                'humidity': hums,
                'dataset_type': dataset_type,
                'labels': labels
            }
            
            global_feature_path = os.path.join(features_save_path, f'global_{dataset_type}_features.pt')
            torch.save(global_save_data, global_feature_path)
            
            # 清理内存
            del features, global_save_data

            if hasattr(self, 'args') and self.args.use_gpu:
                torch.cuda.empty_cache()
            gc.collect()
            
            return features_save_path
            
        except Exception as e:
            print(f"保存全局特征时出错: {e}")
            # 打印更详细的错误信息
            import traceback
            traceback.print_exc()
            return None

    def _build_model(self):
        # 打印可用GPU信息
        if self.args.use_gpu:
            print(f"可见GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # # 在创建模型前设置参数
        # self.args.num_sensors = 8  # 传感器数量
        # self.args.num_env_sensors = 5  # 环境传感器数量
        # self.args.num_wavelets_scales = 3  # 小波尺度数量
        # self.args.pooled_points = 384  # 池化点数

        # 创建模型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 对模型应用混合精度
        if hasattr(self.args, 'use_amp') and self.args.use_amp:
            print("启用混合精度训练 (FP16)")
        
        if self.args.use_gpu and hasattr(self.args, 'use_multi_gpu') and self.args.use_multi_gpu:
            # DataParallel模式
            print(f"警告: 使用DataParallel可能导致OOM")
            print(f"设备IDs: {self.args.device_ids}")
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        else:
            # 单GPU模式
            model = model.to(self.args.device)
            print(f"在单个设备上运行: {self.args.device}")
        
        return model

    def _get_data(self, flag):
        print(f"加载{flag}数据集...")
        
        # 清理GPU缓存
        if self.args.use_gpu:
            torch.cuda.empty_cache()
        
        # 获取数据集
        data_set, data_loader = data_provider(self.args, flag)
        
        print(f"加载了{len(data_set)}个样本，{len(data_loader)}个批次")
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_mode):
        # criterion = nn.MSELoss()
        criterion = Bridge_Loss(loss_mode)
        return criterion
    
    def select_criterion_vali(self, di_mode):
        criterion = Bridge_Loss_Vali(di_mode)
        return criterion 
    


    def vali(self, vali_data, vali_loader, criterion, path):
        """验证函数"""
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_env, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 动态处理不同尺度的小波系数
                wavelet_coeffs = {}
                n_scales = self.args.n_scales if hasattr(self.args, 'n_scales') else 3
                
                # 添加最后一层近似系数和所有细节系数
                wavelet_coeffs[f'cA{n_scales}'] = batch_x[f'cA{n_scales}'].float().to(self.device)
                for scale in range(n_scales, 0, -1):
                    wavelet_coeffs[f'cD{scale}'] = batch_x[f'cD{scale}'].float().to(self.device)
                
                batch_env = batch_env.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                # 处理批次数据 - 验证阶段不需要详细信息
                batch_result = self._process_batch_and_collect_features(
                    batch_x_mark, batch_env, batch_y_mark, criterion,
                    return_details=False, anomaly_threshold=None,
                    **wavelet_coeffs
                )
                
                # 收集损失
                loss = batch_result['loss']
                total_loss.append(loss)

                if (i + 1) % 200 == 0:
                    print("\titers: {0} | loss: {1:.7f}".format(i + 1, loss.mean().item()))

            # 合并所有批次的损失
            total_loss = torch.cat(total_loss, dim=0)

        self.model.train()
        return total_loss

    def train(self, setting):
        # 加载数据
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        
        # 创建目录结构
        print(f"创建目录结构...")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        
        path_train = os.path.join(self.args.train_loss_comparation, setting)
        os.makedirs(path_train, exist_ok=True)
        
        path_vali_loss_save = os.path.join(self.args.loss_save, setting, 'vali')
        os.makedirs(path_vali_loss_save, exist_ok=True)
        
        train_steps = len(train_loader)
        
        # 创建早停和优化器
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss_mode)
        criterion_vali = self.select_criterion_vali(self.args.di_mode)

        # 记录训练数据
        epoch_train_mean = []
        epoch_vali_mean = []
        epoch_train_std = []
        epoch_vali_std = []
        
        # 显示训练信息
        print(f"开始训练: 总批次={train_steps}, 数据集大小={len(train_data)}")
        print(f"训练轮数: {self.args.train_epochs}, 批次大小: {self.args.batch_size}")
        
        improved_count = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_env, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                print(f"训练轮数: {epoch + 1}, 批次: {i + 1}")
                iter_count += 1
                model_optim.zero_grad()
                
                # 动态处理不同尺度的小波系数
                wavelet_coeffs = {}
                n_scales = self.args.n_scales if hasattr(self.args, 'n_scales') else 3  # 默认3个尺度
                
                # 添加最后一层近似系数
                wavelet_coeffs[f'cA{n_scales}'] = batch_x[f'cA{n_scales}'].float().to(self.device)
                
                # 添加所有尺度的细节系数
                for scale in range(n_scales, 0, -1):
                    wavelet_coeffs[f'cD{scale}'] = batch_x[f'cD{scale}'].float().to(self.device)
                
                batch_env = batch_env.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                # 处理批次数据 - 训练阶段不需要详细信息
                batch_result = self._process_batch_and_collect_features(
                    batch_x_mark, batch_env, batch_y_mark, criterion,
                    return_details=False, anomaly_threshold=None, **wavelet_coeffs
                )
                
                # 计算损失并反向传播
                loss = batch_result['loss']
                loss.mean().backward()
                model_optim.step()
                
                train_loss.append(loss)
                
                # 打印信息
                if (i + 1) % 200 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.mean().item()))
                    speed = (time.time() - epoch_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    epoch_time = time.time()
            
            train_loss = torch.stack(train_loss, dim=0)
            train_loss_mean, train_loss_std = percentile_cal(train_loss, per_num=90)
            
            epoch_train_mean.append(train_loss_mean)
            epoch_train_std.append(train_loss_std)
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("start validate!")
            
            # 验证过程
            vali_loss = self.vali(val_data, val_loader, criterion_vali, path_vali_loss_save)
            vali_loss_mean, vali_loss_std = percentile_cal(vali_loss, per_num=90)
            epoch_vali_mean.append(vali_loss_mean)
            epoch_vali_std.append(vali_loss_std)
            
            print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                epoch + 1, train_steps, float(np.mean(train_loss_mean)), float(np.mean(vali_loss_mean))))
            
            # 早停处理
            best_mean, best_std, improved = early_stopping(vali_loss_mean, vali_loss_std, self.model, path, vali_loss)
            if improved:
                improved_count += 1
                if improved_count % 3 == 0:
                    adjust_learning_rate(model_optim, epoch + 1, self.args)
                    print(f"Learning rate adjusted at {improved_count} improvements.")
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            
        # 测试最佳模型
        self.model.train()
        best_model_path = path + '/' + 'checkpoint.pth'
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        plot_loss(epoch_train_std, epoch_vali_std, None, None, path_train, self.args.patience, loss_type='mean')
        plot_loss(epoch_train_std, epoch_vali_std, None, None, path_train, self.args.patience, loss_type='std')
        print("训练和验证loss图已保存!")
        print('训练结束！')
        print('开始测试')
        test_data, test_loader = self._get_data(flag='test')
        test_health_data, test_health_loader = self._get_data(flag='test_health')
        
        # 再次清理内存
        if self.args.use_gpu:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        gc.collect()
        
        # 假设命令行参数 --sensors "3,5,6"
        # print(args.sensors)
        # 处理前: "3,5,6"
        if isinstance(self.args.sensors, str) and self.args.sensors != 'all':
            self.args.sensors = [f"sensor{int(s.strip())}" for s in self.args.sensors.split(',')]
        # print(args.sensors)
        # 处理后: ['sensor3', 'sensor5', 'sensor6']
        # 测试验证集
        self.test(setting, 
                flag_one = 'vali', flag_two = None,
                healthy_data = val_data, healthy_loader = val_loader, 
                damaged_data =  None, damaged_loader = None, 
                test = 1, 
                vali_loss = early_stopping.best_vali if hasattr(early_stopping, 'best_vali') else None,  
                path_train = path_train)
        
        # 测试健康测试集
        self.test(setting, 
                flag_one = 'test_health', flag_two = None,
                healthy_data = test_health_data, healthy_loader = test_health_loader, 
                damaged_data =  None, damaged_loader = None, 
                test = 1, 
                vali_loss = early_stopping.best_vali if hasattr(early_stopping, 'best_vali') else None,  
                path_train = path_train)
        
        # 测试损伤测试集
        self.test(setting, 
                flag_one = 'test', flag_two = None,
                healthy_data = test_data, healthy_loader = test_loader, 
                  damaged_data =  None, damaged_loader = None, 
                  test = 1, 
                vali_loss = early_stopping.best_vali if hasattr(early_stopping, 'best_vali') else None,  
                path_train = path_train)

    def test(self, setting, flag_one, flag_two=None, healthy_data=None, healthy_loader=None,  
             damaged_data=None, damaged_loader=None, test=0, vali_loss=None, path_train=None):
        """测试方法 - 用于验证和测试模型性能"""
        # 设置保存路径
        path_loss_save = os.path.join(self.args.loss_save, setting, flag_one)
        if not os.path.exists(path_loss_save):
            os.makedirs(path_loss_save)
            
        # 选择评估标准
        criterion = self.select_criterion_vali(self.args.di_mode)
        
        # 加载模型
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.vali_loss = checkpoint['val_loss']
            self.vali_std = checkpoint['val_std']
            print('loaded successfully!')
        else:
            self.vali_loss = self.vali_loss_mean
            self.vali_std = self.vali_loss_std
            
        # 设置异常检测阈值
        anomaly_threshold = self.vali_std
        
        # 设置温度分箱参数 - 减少bins数量和每个bin的样本数
        temp_range = (-10, 30, 5)  # 增大步长减少bins数量
        min_temp, max_temp, step = temp_range
        temp_bins = [(t, t + step) for t in range(min_temp, max_temp, step)]
        samples_per_bin = 20 # 减少每个bin的样本数
        
        # 清理GPU内存
        if self.args.use_gpu:
            torch.cuda.empty_cache()
            print(f"初始GPU内存状态: 已分配 {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # 处理验证数据
        if flag_one == 'vali':
            print('>>>>>>>validating data: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            # 初始化数据收集容器
            total_loss = []
            preds = []
            trues = []
            data_stamps = []
            
            # 添加全局特征收集
            global_features_vali = []
            global_temps_vali = []
            global_hums_vali = []
            global_labels_vali = []
            
            # 初始化温度分箱数据
            temp_binned_data = {i: {
                'cA3': [], 'cD3': [], 'cD2': [], 'cD1': [],  # 分别存储不同尺度的数据
                'temp': [], 
                'humidity': [], 
                'spatial_attention': [],  # 每个箱的空间注意力
                'temporal_attention': [], # 每个箱的时间注意力
                'env_attention': [],  # 环境注意力
                'features': []  # 隐藏特征
            } for i in range(len(temp_bins))}
            completed_bins = set()
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 处理每个批次
            with torch.no_grad():
                for i, (batch_x, batch_env, batch_x_mark, batch_y_mark) in enumerate(healthy_loader):
                    # 清理GPU内存
                    if i % 5 == 0 and self.args.use_gpu:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # 数据转换
                    print(f'{i} batch is processing...')
                    
                    # 动态处理不同尺度的小波系数
                    wavelet_coeffs = {}
                    n_scales = self.args.n_scales if hasattr(self.args, 'n_scales') else 3
                    
                    # 添加最后一层近似系数和所有细节系数
                    wavelet_coeffs[f'cA{n_scales}'] = batch_x[f'cA{n_scales}'].float().to(self.device)
                    for scale in range(n_scales, 0, -1):
                        wavelet_coeffs[f'cD{scale}'] = batch_x[f'cD{scale}'].float().to(self.device)
                    
                    batch_env = batch_env.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    
                    # 批次处理 - 测试阶段需要详细信息
                    batch_result = self._process_batch_and_collect_features(
                        batch_x_mark, batch_env, batch_y_mark, criterion,
                        anomaly_threshold=anomaly_threshold, return_details=True,
                        **wavelet_coeffs
                    )
                    
                    # 收集评估数据 - 立即在CPU上处理并释放GPU张量
                    total_loss.append(batch_result['loss'].detach().cpu())
                    data_stamps.append(batch_result['data_stamp'])
                    preds.append(batch_result['pred'])
                    trues.append(batch_result['true'])
                    
                    # 如果是奇数批次,跳过温度处理以节省内存
                    if i % 1 == 0:
                        # 收集温度分箱数据
                        self._process_temp_binning(
                            batch_result, batch_result['temps'], temp_bins, 
                            temp_binned_data, completed_bins,
                            samples_per_bin
                        )
                    
                    # 每10批次收集一次全局特征,减少内存使用
                    if i % 5 == 0:
                        global_features_vali.append(batch_result['hidden_states']['fused'].cpu())
                        global_temps_vali.extend(batch_result['temps'])
                        if batch_result['humidity'] is not None:
                            global_hums_vali.extend(batch_result['humidity'])
                        global_labels_vali.extend(['vali'] * len(batch_result['temps']))
                    
                    # 手动释放batch_result中的大型张量
                    del batch_result
                    # 释放CUDA内存
                    if self.args.use_gpu:
                        torch.cuda.empty_cache()
                    
                    if (i + 1) % 200 == 0:
                        print("\titers: {0} | loss: {1:.7f}".format(i + 1, total_loss[-1].mean().item()))
                        print(f"\tGPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                # 合并所有数据
                try:
                    # preds = torch.cat(preds, dim=0)
                    # trues = torch.cat(trues, dim=0)
                    preds = torch.zeros((0, 288012, 8), dtype=torch.float32)
                    trues = torch.zeros((0, 288012, 8), dtype=torch.float32)
                    total_loss = torch.cat(total_loss, dim=0)
                    data_stamps = np.concatenate([ds.to_numpy() for ds in data_stamps])
                    timestamps = pd.Series(data_stamps.astype(int))
                    
                    # 保存结果
                    result_save(trues, preds, path_loss_save, labels=None, losses=total_loss, 
                               timestamps=timestamps, flag='vali', moe_weight=None,sensor_names = self.args.sensors)
                    print('vali_loss_save is done!')
                except Exception as e:
                    print(f"错误: {e}")
                    print("跳过验证结果保存")
                
                # 清理内存后处理温度分箱数据
                import gc
                if self.args.use_gpu:
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 处理全局特征
                try:
                    # 处理全局特征 - 确保列表非空
                    if global_features_vali:
                        features_save_path = self._save_global_features(
                            global_features_vali, 
                            global_temps_vali, 
                            global_hums_vali, 
                            global_labels_vali, 
                            'vali', 
                            path_loss_save
                        )
                        print('vali_global_features is done!')
                    else:
                        print("全局特征列表为空,跳过保存")
                except Exception as e:
                    print(f"全局特征保存错误: {e}")
                
                # 最终内存清理
                del preds, trues, total_loss, global_features_vali
                if self.args.use_gpu:
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # 处理测试数据（健康和损伤）
        elif flag_one == 'test_health':
            # === 测试健康数据 ===
            print('>>>>>>>testing health data: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            # 初始化数据收集容器
            preds = []
            trues = []
            masks = []
            labels = []
            losses = []
            data_stamps = []
            
            # 添加全局特征收集
            global_features_test_health = []
            global_temps_test_health = []
            global_hums_test_health = []
            global_labels_test_health = []
            
            # 初始化温度分箱数据
            temp_binned_data_health = {i: {
                'cA3': [], 'cD3': [], 'cD2': [], 'cD1': [],  # 分别存储不同尺度的数据
                'temp': [], 
                'humidity': [], 
                'spatial_attention': [],  # 每个箱的空间注意力
                'temporal_attention': [], # 每个箱的时间注意力
                'env_attention': [],  # 环境注意力
                'features': []  # 隐藏特征
            } for i in range(len(temp_bins))}
            completed_bins_health = set()
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 处理每个批次
            with torch.no_grad():
                for i, (batch_x, batch_env, batch_x_mark, batch_y_mark) in enumerate(healthy_loader):
                    # 清理GPU内存
                    if i % 5 == 0 and self.args.use_gpu:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # 数据转换
                    print(f'{i} batch is processing...')
                    
                    # 动态处理不同尺度的小波系数
                    wavelet_coeffs = {}
                    n_scales = self.args.n_scales if hasattr(self.args, 'n_scales') else 3
                    
                    # 添加最后一层近似系数和所有细节系数
                    wavelet_coeffs[f'cA{n_scales}'] = batch_x[f'cA{n_scales}'].float().to(self.device)
                    for scale in range(n_scales, 0, -1):
                        wavelet_coeffs[f'cD{scale}'] = batch_x[f'cD{scale}'].float().to(self.device)
                    
                    batch_env = batch_env.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    
                    # 批次处理 - 测试阶段需要详细信息
                    batch_result = self._process_batch_and_collect_features(
                        batch_x_mark, batch_env, batch_y_mark, criterion,
                        anomaly_threshold=anomaly_threshold, return_details=True,
                        **wavelet_coeffs
                    )
                    print('test_health_batch_result is done!')
                    
                    # 收集评估数据 - 立即在CPU上处理
                    labels.append(batch_result['sample_labels'].cpu() if batch_result['sample_labels'] is not None else None)
                    masks.append(batch_result['mask'])
                    data_stamps.append(batch_result['data_stamp'])
                    preds.append(batch_result['pred'])
                    trues.append(batch_result['true'])
                    losses.append(batch_result['loss'].detach().cpu())
                    
                    # 如果是奇数批次,跳过温度处理以节省内存
                    if i % 1 == 0:
                        # 收集温度分箱数据
                        self._process_temp_binning(
                            batch_result, batch_result['temps'], temp_bins, 
                            temp_binned_data_health, completed_bins_health,
                            samples_per_bin
                        )
                    
                    # 每10批次收集一次全局特征,减少内存使用
                    if i % 5 == 0:
                        global_features_test_health.append(batch_result['hidden_states']['fused'].cpu())
                        global_temps_test_health.extend(batch_result['temps'])
                        if batch_result['humidity'] is not None:
                            global_hums_test_health.extend(batch_result['humidity'])
                        global_labels_test_health.extend(['test_health'] * len(batch_result['temps']))
                
                # 合并所有数据
                # preds = torch.cat(preds, dim=0)
                # trues = torch.cat(trues, dim=0)
                preds = torch.zeros((0, 288012, 8), dtype=torch.float32)
                trues = torch.zeros((0, 288012, 8), dtype=torch.float32)
                losses = torch.cat(losses, dim=0)
                labels = torch.cat(labels, dim=0)
                data_stamps = np.concatenate([ds.to_numpy() for ds in data_stamps])
                timestamps = pd.Series(data_stamps.astype(int))

                
                # 保存结果
                result_save(preds, trues, path_loss_save, labels, losses, timestamps, 
                           flag_one, moe_weight=None,sensor_names = self.args.sensors)
                
                # 保存注意力权重和特征
                attn_save_path = os.path.join(path_loss_save, 'attention_weights')
                os.makedirs(attn_save_path, exist_ok=True)
                
                # 直接保存温度分箱数据
                self.save_binned_data(temp_binned_data_health, 'test_health', temp_bins, attn_save_path)
                print('test_health attention weights save done!')
                
                # 处理全局特征
                features_save_path = self._save_global_features(
                    [torch.cat(global_features_test_health, dim=0)], 
                    global_temps_test_health, 
                    global_hums_test_health, 
                    global_labels_test_health, 
                    'test_health', 
                    path_loss_save
                )
                print('test_health_global_features is done!')
            # === 处理损伤数据 ===
        elif flag_one == 'test':
            print('>>>>>>>testing data: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            path_loss_save_two = os.path.join(self.args.loss_save, setting, flag_one)
            if not os.path.exists(path_loss_save_two):
                os.makedirs(path_loss_save_two)
                
            # 初始化数据收集容器
            preds_two = []
            trues_two = []
            masks_two = []
            labels_two = []
            losses_two = []
            data_stamps_two = []
            
            # 添加全局特征收集
            global_features_test = []
            global_temps_test = []
            global_hums_test = []
            global_labels_test = []
            
            # 初始化温度分箱数据
            temp_binned_data_test = {i: {
                'cA3': [], 'cD3': [], 'cD2': [], 'cD1': [],  # 分别存储不同尺度的数据
                'temp': [], 
                'humidity': [], 
                'spatial_attention': [],  # 每个箱的空间注意力
                'temporal_attention': [], # 每个箱的时间注意力
                'env_attention': [],  # 环境注意力
                'features': []  # 隐藏特征
            } for i in range(len(temp_bins))}
            completed_bins_test = set()
            
            # 处理每个批次
            with torch.no_grad():
                for i, (batch_x, batch_env, batch_x_mark, batch_y_mark) in enumerate(healthy_loader):
                    # 清理GPU内存
                    if i % 3 == 0 and self.args.use_gpu:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # 数据转换
                    print(f'{i} batch is processing...')
                    
                    # 动态处理不同尺度的小波系数
                    wavelet_coeffs = {}
                    n_scales = self.args.n_scales if hasattr(self.args, 'n_scales') else 3
                    
                    # 添加最后一层近似系数和所有细节系数
                    wavelet_coeffs[f'cA{n_scales}'] = batch_x[f'cA{n_scales}'].float().to(self.device)
                    for scale in range(n_scales, 0, -1):
                        wavelet_coeffs[f'cD{scale}'] = batch_x[f'cD{scale}'].float().to(self.device)
                    
                    batch_env = batch_env.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    
                    # 批次处理 - 测试阶段需要详细信息
                    batch_result = self._process_batch_and_collect_features(
                        batch_x_mark, batch_env, batch_y_mark, criterion,
                        anomaly_threshold=anomaly_threshold, return_details=True,
                        **wavelet_coeffs
                    )
                    print(f'test_batch_result {i} is done!')
                    
                    # 收集评估数据 - 立即在CPU上处理
                    labels_two.append(batch_result['sample_labels'].cpu() if batch_result['sample_labels'] is not None else None)
                    # masks_two.append(batch_result['mask'])
                    data_stamps_two.append(batch_result['data_stamp'])
                    # preds_two.append(batch_result['pred'])
                    # trues_two.append(batch_result['true'])
                    losses_two.append(batch_result['loss'].detach().cpu())
                    

                    if i % 1 == 0:
                        # 收集温度分箱数据
                        self._process_temp_binning(
                            batch_result, batch_result['temps'], temp_bins, 
                            temp_binned_data_test, completed_bins_test,
                            samples_per_bin
                        )
                    
                    # 每5次收集一次全局特征,减少内存使用
                    if i % 5 == 0:
                        global_features_test.append(batch_result['hidden_states']['fused'].cpu())
                        global_temps_test.extend(batch_result['temps'])
                        if batch_result['humidity'] is not None:
                            global_hums_test.extend(batch_result['humidity'])
                        global_labels_test.extend(['test'] * len(batch_result['temps']))
                    
                    # 手动释放batch_result中的大型张量
                    del batch_result
                    # 释放CUDA内存
                    if self.args.use_gpu:
                        torch.cuda.empty_cache()
                    
                    if (i + 1) % 10 == 0:
                        print(f"\t已处理 {i + 1} 批次，当前GPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    
                    # 合并所有数据
                # preds_two = torch.cat(preds_two, dim=0)
                # trues_two = torch.cat(trues_two, dim=0)
                losses_two = torch.cat(losses_two, dim=0)
                labels_two = torch.cat(labels_two, dim=0)
                data_stamps_two = np.concatenate([ds.to_numpy() for ds in data_stamps_two])
                timestamps_two = pd.Series(data_stamps_two.astype(int))
                preds_two = torch.zeros((0, 288012, 8), dtype=torch.float32)
                trues_two = torch.zeros((0, 288012, 8), dtype=torch.float32)
                
                # 保存结果
                result_save(preds_two, trues_two, path_loss_save_two, labels_two, losses_two, 
                        timestamps_two, flag_one, moe_weight=None,sensor_names = self.args.sensors)
                print('test_two_loss_save is done!')
                    
                # 保存注意力权重和特征
                attn_save_path = os.path.join(path_loss_save_two, 'attention_weights')
                os.makedirs(attn_save_path, exist_ok=True)
                
                # 直接保存温度分箱数据
                self.save_binned_data(temp_binned_data_test, 'test', temp_bins, attn_save_path)
                print('test attention weights save done!')
                    
                # 处理全局特征
                features_save_path_test = self._save_global_features(
                    [torch.cat(global_features_test, dim=0)], 
                    global_temps_test, 
                    global_hums_test, 
                    global_labels_test, 
                    'test', 
                    path_loss_save_two
                )
            print('test_global_features is done!')
    
        return

    def save_binned_data(self, temp_binned_data, dataset_type, temp_bins, save_path):
        """保存按温度分箱的数据"""
        # 保存每个温度箱的数据
        for bin_idx, (bin_min, bin_max) in enumerate(temp_bins):
            if bin_idx not in temp_binned_data:
                continue
            
            bin_data = temp_binned_data[bin_idx]
            
            # 确保有数据
            if not bin_data['cA3']:
                continue
            
            # 准备保存数据
            save_data = {
                'temperature': np.array(bin_data['temp']),
                'temp_range': (bin_min, bin_max),
                'dataset_type': dataset_type,
                'count': len(bin_data['cA3']),
                # 小波数据
                'wavelet_data': {
                    scale: torch.cat(data, dim=0) 
                    for scale, data in bin_data.items() 
                    if scale in ['cA3', 'cD3', 'cD2', 'cD1']
                },
                # 注意力权重
                'spatial_attention': bin_data['spatial_attention'],
                'temporal_attention': bin_data['temporal_attention'],
                # 'env_attention': bin_data['env_attention'],
                # 特征
                'features': torch.cat(bin_data['features'], dim=0) if bin_data['features'] else None
            }
            
            # 添加湿度数据（如果存在）
            if bin_data['humidity']:
                save_data['humidity'] = np.array(bin_data['humidity'])
            
            # 保存数据
            bin_save_path = os.path.join(save_path, f'bin_{bin_min}_{bin_max}_{dataset_type}.pt')
            torch.save(save_data, bin_save_path)
            