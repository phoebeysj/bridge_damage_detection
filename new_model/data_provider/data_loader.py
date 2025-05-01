import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import inject_sensor_fault
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
import pywt  # 导入PyWavelets库用于小波变换

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.stride = 60000

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def preprocess_z24_data(args, root_path, data_path):
    """预处理Z24数据，只需要在训练开始前调用一次
    
    Args:
        args: 参数对象
        root_path: 根目录路径
        data_path: 数据目录路径
        
    Returns:
        processed_data: 包含处理后数据的字典
    """
    env_columns = ['WS', 'WD', 'H', 'AT', 'TE']
    
    # 数据路径
    train_data_path = 'data_six_months.parquet'
    train_env_path = 'env_data_train.parquet'
    test_health_data_path = 'data_test_health0.parquet'
    test_damage_data_path = 'data_test_damage0.parquet'
    test_env_path = 'data_test_env.parquet'
    
    # 缓存文件路径
    cache_dir = os.path.join(root_path, data_path, 'processed_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # 为按月分组的数据使用不同的缓存文件名
    cache_file = os.path.join(cache_dir, f'processed_data_{args.sensors}_new_part.pt')
    
    # 检查缓存是否存在
    if os.path.exists(cache_file):
        print(f"找到已处理的数据缓存，正在加载: {cache_file}")
        try:
            return torch.load(cache_file)
        except RuntimeError:
            print(f"缓存文件格式不兼容，将重新处理数据...")
            # 继续执行数据处理逻辑
    
    print("未找到数据缓存，开始处理数据...")
    
    # 1. 读取训练数据和健康测试数据
    if args.sensors != 'all':
        if isinstance(args.sensors, str):
            sensors = [f"sensor{int(s.strip())}" for s in args.sensors.split(',')]
        else:
            sensors = args.sensors
        columns = ['datetime'] + sensors
        train_data = pd.read_parquet(
            os.path.join(root_path, data_path, train_data_path),
            columns=columns
        ).fillna(0)
        
        test_health_data = pd.read_parquet(
            os.path.join(root_path, data_path, test_health_data_path),
            columns=columns
        ).fillna(0)
        
        test_damage_data = pd.read_parquet(
            os.path.join(root_path, data_path, test_damage_data_path),
            columns=columns
        ).fillna(0)
    else:
        train_data = pd.read_parquet(
            os.path.join(root_path, data_path, train_data_path)
        ).fillna(0)
        
        test_health_data = pd.read_parquet(
            os.path.join(root_path, data_path, test_health_data_path)
        ).fillna(0)
        
        test_damage_data = pd.read_parquet(
            os.path.join(root_path, data_path, test_damage_data_path)
        ).fillna(0)
    
    # 2. 设置索引并合并数据
    train_data.set_index('datetime', inplace=True)
    test_health_data.set_index('datetime', inplace=True)
    test_damage_data.set_index('datetime', inplace=True)
    
    # 合并数据集（健康数据）
    combined_data = pd.concat([train_data, test_health_data])
    combined_data = combined_data.sort_index()  # 确保按时间顺序
    
    # 3. 缩放处理
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_path = os.path.join(root_path, data_path, 'scaler_params.npz')
    
    # 使用所有健康数据拟合缩放器
    scaler.fit(combined_data.values)
    np.savez(scaler_path, 
            scale_=scaler.scale_,
            min_=scaler.min_,
            data_min_=scaler.data_min_,
            data_max_=scaler.data_max_,
            data_range_=scaler.data_range_)
    
    # 4. 读取环境数据
    train_env = pd.read_parquet(
        os.path.join(root_path, data_path, train_env_path),
        columns=env_columns
    ).fillna(0)
    
    test_env = pd.read_parquet(
        os.path.join(root_path, data_path, test_env_path),
        columns=env_columns
    ).fillna(0)
    
    # 分离测试环境数据
    test_env_health = test_env[test_env.index < pd.to_datetime('1998-08-01')]
    test_env_damage = test_env[test_env.index > pd.to_datetime('1998-08-01')]
    
    # 合并健康环境数据
    combined_env = pd.concat([train_env, test_env_health]).sort_index()
    
    # 5. 按月分组划分数据
    combined_data['month'] = combined_data.index.month
    combined_data['year'] = combined_data.index.year
    
    # 按年月分组 - 传感器数据按相同方式分组
    monthly_groups = combined_data.groupby(['year', 'month'])
    
    # 创建月份索引字典，存储每个月份数据的元数据
    train_month_indices = {}
    val_month_indices = {}
    test_health_month_indices = {}
    
    # 记录跳过的月份
    skipped_months = []
    
    # 对每个月的数据进行分割：70% 训练, 15% 验证, 15% 测试健康
    for (year, month), group in monthly_groups:
        # 移除年月列
        group = group.drop(['month', 'year'], axis=1)
        
        # 分割数据
        n = len(group)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        # 分割传感器数据
        train_month = group.iloc[:train_end]
        val_month = group.iloc[train_end:val_end]
        test_health_month = group.iloc[val_end:]
        
        # 收集月份信息
        month_key = f"{year}-{month:02d}"
        
        # 存储每个月份的数据及其索引信息
        train_month_indices[month_key] = {
            'data': scaler.transform(train_month.values),
            'timestamps': train_month.index,
            'length': len(train_month)
        }
        
        val_month_indices[month_key] = {
            'data': scaler.transform(val_month.values),
            'timestamps': val_month.index,
            'length': len(val_month)
        }
        
        test_health_month_indices[month_key] = {
            'data': scaler.transform(test_health_month.values),
            'timestamps': test_health_month.index, 
            'length': len(test_health_month)
        }
    
    # 处理所有数据
    processed_data = {
        'train_data': {
            'month_indices': train_month_indices  # 只保存按月索引，不再保存完整数据
        },
        'val_data': {
            'month_indices': val_month_indices  # 只保存按月索引，不再保存完整数据
        },
        'test_health_data': {
            'month_indices': test_health_month_indices  # 只保存按月索引，不再保存完整数据
        },
        'test_data': {
            'data_x': scaler.transform(test_damage_data.values),
            'timestamps': test_damage_data.index,
            'env_data': test_env_damage
        },
        'env_data': combined_env,
        'scaler': scaler
    }
    
    # 保存处理后的数据
    torch.save(processed_data, cache_file, pickle_protocol=4)
    print(f"数据处理完成并已保存到: {cache_file}")
    
    return processed_data

class Z24Dataset(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='Z24.csv',
                 target='', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        self.seq_len = args.seq_len
        self.label_len = 0
        self.pred_len = args.seq_len
        assert flag in ['train', 'test', 'val', 'test_health']
        type_map = {'train': 0, 'val': 1, 'test_health': 2, 'test': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.env_columns = ['WS', 'WD', 'H', 'AT', 'TE']
        
        # Define data paths
        self.train_data_path = 'data_six_months.parquet'
        self.train_env_path = 'env_data_final.parquet'
        self.test_health_data_path = 'data_test_health0.parquet'
        self.test_damage_data_path = 'data_test_damage0.parquet'
        self.test_env_path = 'data_test_env.parquet'
        
        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Path for saving scaler parameters
        self.scaler_path = os.path.join(root_path, self.data_path, 'scaler_params.npz')
        # 使用传入的stride或适当的默认值（增大默认值以减少窗口数量）
        self.stride = 1000 
        # 每个月最大窗口数量限制
        self.max_windows_per_month = args.max_windows_per_month if hasattr(args, 'max_windows_per_month') else 500
        
        self.__read_data__()
        self.__build_month_windows__()  # 构建月内滑动窗口

    def __read_data__(self):
        """使用预处理好的数据"""
        # 获取预处理数据
        self.processed_data = preprocess_z24_data(self.args, self.root_path, self.data_path)
        
        # 根据flag选择相应的数据
        if self.flag == 'train':
            self.month_data = self.processed_data['train_data']['month_indices']
            # 为了兼容原有代码，从月份数据中重建完整数据集
            self.__rebuild_full_data('train')
        elif self.flag == 'val':
            self.month_data = self.processed_data['val_data']['month_indices']
            # 为了兼容原有代码，从月份数据中重建完整数据集
            self.__rebuild_full_data('val')
        elif self.flag == 'test_health':
            self.month_data = self.processed_data['test_health_data']['month_indices']
            # 为了兼容原有代码，从月份数据中重建完整数据集
            self.__rebuild_full_data('test_health')
        elif self.flag == 'test':
            self.data_x = self.processed_data['test_data']['data_x']
            self.timestamps = self.processed_data['test_data']['timestamps']   
            self.data_env = self.processed_data['test_data']['env_data']
            # 损伤数据不按月索引，因此没有month_data
            
        # 保存环境数据和scaler
        self.combined_env = self.processed_data['env_data']
        self.scaler = self.processed_data['scaler']
        
        print(f"数据集 '{self.flag}' 加载完成, 形状: {self.data_x.shape if hasattr(self, 'data_x') else 'N/A'}")
        print(f"时间范围: {self.timestamps.min() if hasattr(self, 'timestamps') else 'N/A'} 到 {self.timestamps.max() if hasattr(self, 'timestamps') else 'N/A'}")
        print(f"月份数量: {len(self.month_data) if hasattr(self, 'month_data') else 'N/A'}")

    def __rebuild_full_data(self, flag):
        """从月份数据中重建完整数据集，用于兼容原有代码"""
        if not hasattr(self, 'month_data') or not self.month_data:
            print(f"警告: {flag}数据集没有月份数据")
            self.data_x = np.zeros((0, 8))  # 假设有8个传感器
            self.timestamps = pd.DatetimeIndex([])
            return
            
        all_data = []
        all_timestamps = []
        
        # 收集所有月份的数据
        for month_key, month_info in self.month_data.items():
            all_data.append(month_info['data'])
            all_timestamps.append(month_info['timestamps'])
            
        # 合并数据
        if all_data:
            self.data_x = np.vstack(all_data)
            self.timestamps = pd.DatetimeIndex(np.concatenate(all_timestamps))
        else:
            print(f"警告: {flag}数据集没有有效数据")
            self.data_x = np.zeros((0, 8))  # 假设有8个传感器
            self.timestamps = pd.DatetimeIndex([])

    def __build_month_windows__(self):
        """构建月内滑动窗口"""
        self.valid_windows = []
        
        # 处理损伤测试数据（不按月划分）
        if self.flag == 'test':
            # 对于损伤数据，使用常规滑动窗口
            for i in range(0, len(self.data_x) - self.seq_len + 1, self.stride):
                self.valid_windows.append((i, i + self.seq_len))
            print(f"测试数据: 构建了 {len(self.valid_windows)} 个滑动窗口")
            return
            
        # 处理健康数据（按月划分）
        skipped_months = []
        for month_key, month_info in self.month_data.items():
            month_data = month_info['data']
            month_length = month_info['length']
            
            # 如果这个月的数据量不足，跳过
            if month_length < self.seq_len:
                skipped_months.append(month_key)
                continue
                
            # 为该月创建滑动窗口索引
            month_windows = 0
            for i in range(0, month_length - self.seq_len + 1, self.stride):
                window_start = i
                window_end = i + self.seq_len
                self.valid_windows.append((month_key, window_start, window_end))
                month_windows += 1
                
            print(f"月份 {month_key}: 构建了 {month_windows} 个滑动窗口")
                
        if skipped_months:
            print(f"警告: 跳过了 {len(skipped_months)} 个月份，因为数据量不足: {', '.join(skipped_months)}")
            
        print(f"{self.flag}数据集: 总共构建了 {len(self.valid_windows)} 个月内滑动窗口")

    def wavelet_transform_concat(self, data):
        """
        对数据进行小波变换，返回各层系数
        
        Args:
            data (np.ndarray): Shape (seq_len, n_sensors) 的输入数据
            
        Returns:
            dict: 包含不同尺度小波系数的字典，键名格式为：
                - cA{n_scales}: 最后一层近似系数
                - cD{i}: 第i层细节系数，其中i从1到n_scales
        """
        import pywt
        import numpy as np
        
        # 获取传感器数量和小波分解层数
        n_sensors = data.shape[1]
        n_scales = self.args.n_scales if hasattr(self.args, 'n_scales') else 1
        
        # 初始化存储各层系数的字典
        coeffs_dict = {f'cA{n_scales}': [], f'cD{n_scales}': []}
        for i in range(n_scales-1, 0, -1):
            coeffs_dict[f'cD{i}'] = []
        
        # 对每个传感器分别进行小波变换
        for i in range(n_sensors):
            sensor_data = data[:, i]
            # 进行n_scales层小波分解
            coeffs = pywt.wavedec(sensor_data, 'db4', level=n_scales)
            
            # 收集各层系数
            for j, coeff in enumerate(coeffs):
                if j == 0:
                    coeffs_dict[f'cA{n_scales}'].append(coeff)
                else:
                    coeffs_dict[f'cD{n_scales-j+1}'].append(coeff)
        
        # 将同尺度的系数转换为tensor，shape为 [coeff_length, n_sensors]
        result_dict = {}
        for k, v in coeffs_dict.items():
            result_dict[k] = torch.FloatTensor(np.array(v).T)
        
        return result_dict
    def  env_embedding(self, env_data):
        """
        对环境数据进行embedding
        # 使用预训练的embedding模型对环境数据进行embedding
        #  环境数据编码为二进制
        """
        return env_data
        
        
    def __getitem__(self, index):
        # 检查索引是否有效
        if index >= len(self.valid_windows):
            index = index % len(self.valid_windows)
        
        if self.flag == 'test':
            # 处理测试（损伤）数据
            s_begin, s_end = self.valid_windows[index]
            seq_x = self.data_x[s_begin:s_end]
            current_timestamp = self.timestamps[s_begin]
        else:
            # 处理健康数据
            try:
                month_key, window_start, window_end = self.valid_windows[index]
                seq_x = self.month_data[month_key]['data'][window_start:window_end]
                current_timestamp = self.month_data[month_key]['timestamps'][window_start]
            except (KeyError, IndexError) as e:
                print(f"错误: 获取月份数据时出错: {e}, 索引={index}, 窗口数={len(self.valid_windows)}")
                seq_x = np.zeros((self.seq_len, 8))
                current_timestamp = pd.Timestamp.now()
        
        # 进行小波变换，获取字典格式的系数
        wavelet_coeffs = self.wavelet_transform_concat(seq_x)
        
        # 处理环境数据
        closest_timestamp = pd.Timestamp(current_timestamp).floor('H')
        env_data_source = self.data_env if self.flag == 'test' else self.combined_env
        available_timestamps = env_data_source.index
        closest_idx = available_timestamps.searchsorted(closest_timestamp)
        if closest_idx >= len(available_timestamps):
            closest_idx = len(available_timestamps) - 1
        
        hours_needed = self.args.hours_needed
        end_idx = min(closest_idx + hours_needed, len(available_timestamps))
        env_timestamps = available_timestamps[closest_idx:end_idx]
        seq_env = torch.FloatTensor(env_data_source.loc[env_timestamps].values)
        # seq_env = self.env_embedding(seq_env)
        # 处理时间特征
        if self.flag == 'test':
            timestamps = self.timestamps[s_begin:s_end]
        else:
            timestamps = self.month_data[month_key]['timestamps'][window_start:window_end]
        
        time_features = np.stack([
            timestamps.hour,
            timestamps.day,
            timestamps.dayofweek,
            timestamps.month,
        ], axis=1).astype(np.float32)
        seq_x_mark = torch.FloatTensor(time_features)
        seq_y_mark = torch.FloatTensor(timestamps.astype(np.int64).values)
        
        # 返回字典格式的小波系数和其他特征
        return wavelet_coeffs, seq_env, seq_x_mark, seq_y_mark

    def __len__(self):
        if not hasattr(self, 'valid_windows'):
            return 0
        return len(self.valid_windows)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)