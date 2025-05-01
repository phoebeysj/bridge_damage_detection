import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class DomainSpecificAttention(nn.Module):
    """针对不同域设计的注意力机制"""
    def __init__(self, d_model, n_heads, domain_type):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.domain_type = domain_type
        
        # 每个域有自己的投影矩阵
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = q
        
        batch_size = q.size(0)
        
        # 投影
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.out_proj(context)
        
        # 同时返回注意力权重用于可视化
        return output, attn

class TemporalProcessor(nn.Module):
    """时域处理模块"""
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        
        # 多尺度时间卷积，保持输出通道数相同
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(configs.d_model, configs.d_model, k, padding=k//2)
            for k in [3, 7, 11]  # 不同尺度的感受野
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Linear(configs.d_model * 3, configs.d_model)
        
        # 时域注意力
        self.temporal_attn = DomainSpecificAttention(configs.d_model, configs.n_heads, 'temporal')
        
        self.norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x):
        # 多尺度卷积特征
        x_t = x.transpose(1, 2)  # [batch, d_model, seq_len]
        conv_outs = []
        for conv in self.conv_layers:
            conv_outs.append(conv(x_t))
        
        # 将不同尺度的特征拼接并融合
        conv_out = torch.cat(conv_outs, dim=1)  # [batch, d_model*3, seq_len]
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, d_model*3]
        conv_out = self.feature_fusion(conv_out)  # [batch, seq_len, d_model]
        
        # 时域注意力
        temporal_out, temp_attn = self.temporal_attn(conv_out)
        
        return self.norm(temporal_out + conv_out), temp_attn  # 返回输出和注意力权重

class SpatialDomainAttention(nn.Module):
    """专门处理空间域的注意力机制，保留批次和序列维度"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        # 投影矩阵
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, q, k=None, v=None, mask=None):
        """
        输入:
            q: [batch_size, seq_len, num_sensors, d_model]
            k, v: 同上 (如果未提供则等于q)
        
        输出:
            output: [batch_size, seq_len, num_sensors, d_model]
            attn: [batch_size, seq_len, num_sensors, num_sensors]
        """
        if k is None: k = q
        if v is None: v = q
        
        batch_size, seq_len, num_sensors, d_model = q.size()
        
        # 投影并重塑为(batch_size, seq_len, num_sensors, n_heads, d_k)
        q = self.q_proj(q).view(batch_size, seq_len, num_sensors, self.n_heads, self.d_k)
        k = self.k_proj(k).view(batch_size, seq_len, num_sensors, self.n_heads, self.d_k)
        v = self.v_proj(v).view(batch_size, seq_len, num_sensors, self.n_heads, self.d_k)
        
        # 调整维度顺序为(batch_size, seq_len, n_heads, num_sensors, d_k)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)  # [B, L, H, S, S]
        
        context = torch.matmul(attn, v)  # [B, L, H, S, d_k]
        
        # 重塑回原始维度顺序
        context = context.permute(0, 1, 3, 2, 4).contiguous()  # [B, L, S, H, d_k]
        context = context.view(batch_size, seq_len, num_sensors, -1)  # [B, L, S, D]
        
        output = self.out_proj(context)
        
        # 对头维度取平均，得到最终注意力权重
        attn_avg = attn.mean(dim=2)  # [B, L, S, S]
        
        return output, attn_avg

class UnifiedTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.n_heads = configs.n_heads
        self.num_sensors = configs.enc_in
        
        # 时域处理
        self.temporal_processor = TemporalProcessor(configs)
        
        # 空间注意力
        self.spatial_attn = SpatialDomainAttention(configs.d_model, configs.n_heads)
        
        # 环境特征处理 - 简化版本
        self.env_proj = nn.Linear(1, configs.d_model)  # 单个环境传感器投影
        self.env_attn = DomainSpecificAttention(configs.d_model, configs.n_heads, 'environment')
        
        # 特征整合
        self.feature_integration = nn.Sequential(
            nn.Linear(configs.d_model * 3, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model)
        )
        
        self.norm = nn.LayerNorm(configs.d_model)
        
        # 存储注意力权重
        self.attention_weights = {
            'temporal': None,
            'spatial': None,
            'environmental': None
        }
        
    def forward(self, x, x_mark, x_env=None):
        """
        输入:
        x: [batch_size, seq_len, d_model] - 已经过DataEmbedding的传感器数据
        x_mark: [batch_size, seq_len, time_features] - 时间特征
        x_env: [batch_size, num_env_sensors] - 环境数据
        """
        batch_size, seq_len, _ = x.shape
        
        # 时域处理，获取时域注意力
        temporal_features, temp_attn = self.temporal_processor(x)  # [B, L, D], [B*S, H, T, T]
        self.attention_weights['temporal'] = temp_attn
        
        # 空间注意力处理
        batch_size, seq_len, d_model = temporal_features.shape
        
        # 在空间维度上重塑输入
        spatial_input = temporal_features.view(batch_size, seq_len, 1, d_model)
        spatial_input = spatial_input.expand(-1, -1, self.num_sensors, -1)  # [B, L, S, D]
        
        # 使用新的空间注意力
        spatial_features, spatial_attn = self.spatial_attn(spatial_input)  # [B, L, S, D], [B, L, S, S]
        self.attention_weights['spatial'] = spatial_attn
        
        # 展平输出用于后续处理
        spatial_features = spatial_features.view(batch_size, seq_len, self.num_sensors * d_model)
        
        if x_env is not None:
            # 环境数据处理
            batch_size, num_env_sensors = x_env.shape
            
            # 重塑并投影
            x_env = x_env.view(batch_size, num_env_sensors, 1)  # [B, E, 1]
            env_features = self.env_proj(x_env)  # [B, E, D]
            
            # 将环境特征展平为序列
            env_seq = env_features.view(batch_size, -1, self.d_model)  # [B, E, D]
            
            # 使用传感器数据作为查询，环境特征作为键值
            temp_seq = temporal_features.reshape(batch_size, -1, self.d_model)  # [B, L, D]
            
            # 使用env_attn而不是spatial_attn
            _, env_attn = self.env_attn(temp_seq, env_seq, env_seq)  # [B, H, L, E]
            self.attention_weights['environmental'] = env_attn
            
            # 与传感器特征融合
            env_fused, _ = self.env_attn(temporal_features, env_features.unsqueeze(1).expand(-1, seq_len, -1, -1).view(batch_size, seq_len, -1), env_features.unsqueeze(1).expand(-1, seq_len, -1, -1).view(batch_size, seq_len, -1))
            
            # 为特征融合准备数据
            env_features_for_fusion = env_fused  # [B, L, D]
        else:
            env_features_for_fusion = torch.zeros_like(temporal_features)  # [B, L, D]
        
        # 4. 特征整合
        all_features = torch.cat([temporal_features, spatial_features, env_features_for_fusion], dim=-1)  # [B, L, D*3]
        output = self.feature_integration(all_features)  # [B, L, D]
        
        return self.norm(output + temporal_features)  # 残差连接

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 传感器数量
        self.env_dim = configs.env_dim if hasattr(configs, 'env_dim') else 48  # 环境变量数量
        
        # 数据嵌入
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Transformer模块
        self.transformer = UnifiedTransformer(configs)
        
        # 输出投影
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
 

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, x_env):
        """
        输入:
        x_enc: [batch_size, seq_len, num_sensors]
        x_mark_enc: [batch_size, seq_len, time_features]
        x_env: [batch_size, num_env_sensors]
        """
        # 数据嵌入 (包含了位置编码和时间特征)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, D]
            
        # Transformer处理
        transformer_out = self.transformer(enc_out, x_mark_enc, x_env)  # [B, L, D]
        
        # 输出投影
        dec_out = self.projection(transformer_out)  # [B, L, num_sensors]
        return dec_out
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_env=None):
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, x_env)
            return dec_out
        return None 
        
    def forward_with_attention(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_env=None):
        """执行前向传播并返回输出、隐藏状态和注意力权重
        
        返回:
            outputs: 模型的预测输出
            hidden_states: 融合后的特征表示
            attention_weights: 注意力权重
        """
        # 数据嵌入 (包含了位置编码和时间特征)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, D]
            
        # Transformer处理，获取隐藏状态和注意力权重
        transformer_out = self.transformer(enc_out, x_mark_enc, x_env)  # [B, L, D]
        
        # 获取注意力权重
        attention_weights = self.transformer.attention_weights
        
        # 输出投影
        dec_out = self.projection(transformer_out)  # [B, L, num_sensors]
        
        # 返回三个值：输出、隐藏状态（transformer_out）和注意力权重
        return dec_out, transformer_out, attention_weights 