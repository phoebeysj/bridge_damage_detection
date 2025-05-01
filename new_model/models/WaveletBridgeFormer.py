import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from torch.nn.modules.transformer import _get_activation_fn


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionTransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer with attention weights output"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, norm_first=False):
        super().__init__()
        self.self_attn = MultiheadAttentionWithWeights(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # 实现与标准TransformerEncoderLayer相同的组件
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=True):
        """
        Pass the input through the encoder layer.
        
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_attention: whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights) if return_attention=True,
            otherwise just output
        """
        x = src
        
        if self.norm_first:
            # 先进行归一化
            attn_output, attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + attn_output
            x = x + self._ff_block(self.norm2(x))
        else:
            # 后进行归一化
            attn_output, attn_weights = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self._ff_block(x))
        
        if return_attention:
            return x, attn_weights
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block"""
        return self.self_attn(x, x, x, attn_mask=attn_mask, 
                              key_padding_mask=key_padding_mask, 
                              need_weights=True)

    def _ff_block(self, x):
        """Feed forward block"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AttentionTransformerEncoder(nn.Module):
    """TransformerEncoder that can return attention weights"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 制作多个编码器层
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, return_attention=True):
        """
        Args:
            src: 输入序列
            mask: 可选掩码
            src_key_padding_mask: 可选填充掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 变换后的序列
            attention_weights: 所有层的注意力权重 (如果return_attention=True)
        """
        output = src
        attention_weights = []

        for mod in self.layers:
            if return_attention:
                output, attn = mod(output, src_mask=mask, 
                                   src_key_padding_mask=src_key_padding_mask,
                                   return_attention=True)
                attention_weights.append(attn)
            else:
                output = mod(output, src_mask=mask, 
                            src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        if return_attention:
            return output, attention_weights
        return output


class MultiheadAttentionWithWeights(nn.MultiheadAttention):
    """扩展MultiheadAttention类使其总是返回注意力权重"""
    def __init__(self, embed_dim, num_heads, dropout=0., batch_first=False, **kwargs):
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=batch_first, **kwargs)
        
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True):
        # 总是返回注意力权重
        attn_output, attn_output_weights = super().forward(
            query, key, value, key_padding_mask=key_padding_mask,
            need_weights=True, attn_mask=attn_mask, 
            average_attn_weights=average_attn_weights
        )
        return attn_output, attn_output_weights


class SpatialAttention(nn.Module):
    """空间注意力模块：捕捉传感器间的空间关系"""
    def __init__(self, num_sensors, d_model, nhead, dropout=0.1, e_layers=1, n_scales=3, pooled_points=384):
        super(SpatialAttention, self).__init__()
        self.n_scales = n_scales
        self.pooled_points = pooled_points
        # 使用正确的输入维度
        self.input_projection = nn.Linear((self.n_scales + 1) * self.pooled_points, d_model)
        
        # 使用自定义的支持注意力权重返回的编码器层
        encoder_layer = AttentionTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = AttentionTransformerEncoder(encoder_layer, num_layers=e_layers)
        
        self.output_projection = nn.Linear(d_model, (self.n_scales + 1) * self.pooled_points)
        self.norm = nn.LayerNorm((self.n_scales + 1) * self.pooled_points)

    def forward(self, x, return_attention=True):
        # 输入 x: [batch_size, seq_len, num_sensors]
        batch_size, seq_len, num_sensors = x.shape
        
        # 将注意力应用于序列维度，保持传感器维度
        # 每个传感器独立投影
        x = x.transpose(1, 2)# [batch_size, num_sensors, seq_len]
        projected_x = self.input_projection(x)  # [batch_size, num_sensors, d_model]
        
        # 应用自注意力，处理序列维度
        if return_attention:
            attended, attention_weights = self.transformer_encoder(
                projected_x, return_attention=True
            )
        else:
            attended = self.transformer_encoder(projected_x)
        
        # 将注意力结果进行线性投影
        output = self.output_projection(attended) # [batch_size, num_sensors, seq_len]
        
        normalized = self.norm(output)  # 对d_model维度进行norm
        output = normalized.transpose(1, 2)  # 转置回 [B, seq_len, num_sensors]
        
        if return_attention:
            # print(attention_weights[0].shape)
            return output, attention_weights
        return output


class TemporalAttention(nn.Module):
    """时间注意力模块：捕捉时间序列依赖"""
    def __init__(self, num_sensors, d_model, nhead, dropout=0.1, e_layers=1, n_scales=1, pooled_points=384):
        super(TemporalAttention, self).__init__()
        self.n_scales = n_scales
        self.pooled_points = pooled_points
        self.input_projection = nn.Linear(num_sensors, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # self.output_projection = nn.Linear(d_model, num_sensors)
        # 使用自定义的支持注意力权重返回的编码器层
        encoder_layer = AttentionTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = AttentionTransformerEncoder(encoder_layer, num_layers=e_layers)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=True):
        # 输入 x: [batch_size, seq_len, num_sensors]
        # 映射到 [batch_size, seq_len, d_model]
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 应用自注意力，处理时间维度
        if return_attention:
            output, attention_weights = self.transformer_encoder(
                x, return_attention=True
            )
            output = self.norm(output)
            return output, attention_weights
        else:
            output = self.transformer_encoder(x)
            output = self.norm(output)
            return output


class EnvironmentalAttention(nn.Module):
    """环境注意力模块：融合环境数据和振动数据"""
    def __init__(self, num_sensors, num_env_sensors, d_model, seq_len, nhead, dropout=0.1, d_layers=1, n_scales=1, pooled_points=384):
        super(EnvironmentalAttention, self).__init__()
        self.n_scales = n_scales
        self.pooled_points = pooled_points
        
        # 振动数据投影
        self.vib_projection = nn.Linear(num_sensors, d_model)
        
        # 环境数据投影
        self.env_projection = nn.Linear(num_env_sensors, d_model)
        self.env_pos_encoder = PositionalEncoding(d_model)
        
        # 环境数据自注意力
        env_encoder_layer = AttentionTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.env_transformer = AttentionTransformerEncoder(env_encoder_layer, num_layers=d_layers)
        
        # 交叉注意力
        decoder_layer = AttentionTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.cross_transformer = AttentionTransformerDecoder(decoder_layer, num_layers=d_layers)
        # self.output_projection = nn.Linear(d_model, num_sensors)
        # 输出归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, vib_data, env_data, return_attention=True):
        # 振动数据: [batch_size, seq_len, num_sensors]
        # 环境数据: [batch_size, env_seq_len, num_env_sensors]
        
        # 投影振动数据
        vib_projected = self.vib_projection(vib_data)  # [batch_size, seq_len, d_model]
        
        # 投影并处理环境数据
        env_projected = self.env_projection(env_data)  # [batch_size, env_seq_len, d_model]
        env_projected = self.env_pos_encoder(env_projected)
        
        if return_attention:
            env_attended, env_self_attn = self.env_transformer(
                env_projected, return_attention=True
            )
            cross_attended, cross_attn = self.cross_transformer(
                vib_projected, env_attended, return_attention=True
            )
            
            # 合并注意力权重
            attn_weights = {
                'env_self_attention': env_self_attn,
                'cross_attention': cross_attn['cross_attention']
            }
            
            return self.norm(cross_attended), attn_weights
        else:
            env_attended = self.env_transformer(env_projected)
            cross_attended = self.cross_transformer(vib_projected, env_attended)
            return self.norm(cross_attended)


class FusionLayer(nn.Module):
    """融合层：整合空间、时间和环境注意力输出"""
    def __init__(self, d_model, num_sensors):
        super(FusionLayer, self).__init__()
        self.projection_spatial = nn.Linear(num_sensors, d_model)
        self.projection_temporal = nn.Linear(num_sensors, d_model)
        self.projection_env = nn.Linear(num_sensors, d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spatial_output, temporal_output, env_output):
        # 空间输出: [batch_size, seq_len, d_model]
        # 时间输出: [batch_size, seq_len, d_model]
        # 环境输出: [batch_size, seq_len, d_model]
        spatial_output = self.projection_spatial(spatial_output)
       # temporal_output = self.projection_temporal(temporal_output)
        # env_output = self.projection_env(env_output)    
        # 拼接三个输出
        concatenated = torch.cat([spatial_output, temporal_output, env_output], dim=2)
        
        # 融合
        output = self.fusion(concatenated)
        
        return self.norm(output)


class WaveletBridgeFormer(nn.Module):
    """基于小波分解和注意力机制的结构健康监测模型"""
    def __init__(self, args):
        super(WaveletBridgeFormer, self).__init__()
        # 从args中获取固定超参数
        self.d_model = args.d_model  # 注意力隐藏维度(D)
        self.nhead = args.n_heads  # 多头注意力头数(H)
        self.dropout = args.dropout
        self.pooled_points = args.pooled_points  # 现在直接从args获取
        self.e_layers = args.e_layers if hasattr(args, 'e_layers') else 1
        self.d_layers = args.d_layers if hasattr(args, 'd_layers') else 1
        self.env_seq_len = args.env_seq_len if hasattr(args, 'env_seq_len') else 24
        self.n_scales = args.n_scales if hasattr(args, 'n_scales') else 1
        
        
        self.num_sensors = args.num_sensors
        self.num_env_sensors = args.num_env_sensors
        self.num_wavelets_scales = args.num_wavelets_scales
        
        # 初始化自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.pooled_points)  # 1D池化，目标长度为384
        
        # 初始化注意力模块
        self.spatial_attention = SpatialAttention(
            self.num_sensors, self.d_model, self.nhead, self.dropout, self.e_layers,
            n_scales=self.num_wavelets_scales, pooled_points=self.pooled_points
        )
        self.temporal_attention = TemporalAttention(
            self.num_sensors, self.d_model, self.nhead, self.dropout, self.e_layers,
            n_scales=self.num_wavelets_scales, pooled_points=self.pooled_points
        )
        self.env_attention = EnvironmentalAttention(
            self.num_sensors, self.num_env_sensors, self.d_model, 
            self.env_seq_len, self.nhead, self.dropout, self.d_layers,
            n_scales=self.num_wavelets_scales, pooled_points=self.pooled_points
        )
        
        # 初始化融合层
        self.fusion = FusionLayer(self.d_model, self.num_sensors)  # 使用d_model作为输出维度
        
        # 可选：添加最终输出投影层从d_model映射回num_sensors
        self.final_projection = nn.Linear(self.d_model, self.num_sensors)
        
        # 不再需要initialized标志
        print(f"模型初始化完成 - 传感器: {self.num_sensors}, 环境传感器: {self.num_env_sensors}, 小波尺度: {self.num_wavelets_scales}")

    # def forward(self, batch_x, batch_x_mark, decoder_input, batch_y_mark, mask=None, batch_env=None):
    #     """前向传播函数
    #     Args:
    #         batch_x: 原始振动数据 [B, L, N]
    #         batch_x_mark: 小波系数 [B, S, C, N]
    #         decoder_input: 解码器输入 (未使用)
    #         batch_y_mark: 目标时间特征 (未使用)
    #         mask: 掩码 (未使用)
    #         batch_env: 环境数据 [B, T, M]
    #     """
    #     # 提取小波系数 [B, S, C, N]
    #     wavelet_coeffs = batch_x
    #     original_shape = wavelet_coeffs.shape  # 保存原始形状 [B, S, C, N]
    #     original_seq_len = original_shape[2]   # 原始序列长度C
        
    #     # 检查环境数据
    #     if batch_env is None:
    #         raise ValueError("环境数据不能为空")
            
    #     # 1. 自适应池化: [B, S, C, N] -> [B, S, pooled_points, N]
    #     B, S, C, N = wavelet_coeffs.shape
        
    #     # 重组形状以便应用1D池化
    #     # [B, S, C, N] -> [B*S*N, C] -> [B*S*N, 1, C]
    #     reshaped = wavelet_coeffs.permute(0, 1, 3, 2).reshape(B*S*N, C)
    #     reshaped = reshaped.unsqueeze(1)
        
    #     # 应用1D池化
    #     # [B*S*N, 1, C] -> [B*S*N, 1, pooled_points]
    #     pooled = self.adaptive_pool(reshaped)
        
    #     # 恢复原始形状
    #     # [B*S*N, 1, pooled_points] -> [B, S, N, pooled_points] -> [B, S, pooled_points, N]
    #     pooled = pooled.squeeze(1).reshape(B, S, N, self.pooled_points)
    #     pooled = pooled.permute(0, 1, 3, 2)
        
    #     # 重塑为 [B, S*pooled_points, N]
    #     reshaped = pooled.reshape(B, S * self.pooled_points, N)
        
    #     # 2. 空间注意力
    #     spatial_output, spatial_attn = self.spatial_attention(reshaped, return_attention=True)  # [B, S*P, D]
        
    #     # 3. 时间注意力
    #     temporal_output, temporal_attn = self.temporal_attention(spatial_output, return_attention=True)  # [B, S*P, D]
        
    #     # 4. 环境注意力
    #     env_output, env_attn_dict = self.env_attention(reshaped, batch_env, return_attention=True)  # [B, S*P, D]
        
    #     # 5. 融合层
    #     output = self.fusion(spatial_output, temporal_output, env_output)  # [B, S*P, D]
        
    #     # 6. 最终投影
    #     output = self.final_projection(output)  # [B, S*P, N]
        
    #     # 7. 重塑为便于上采样的格式
    #     output = output.reshape(B, S, self.pooled_points, N)
        
    #     # 8. 对每个batch、每个scale、每个传感器进行上采样
    #     full_size_output = torch.zeros((B, S, original_seq_len, N), device=output.device)
        
    #     for b in range(B):
    #         for s in range(S):
    #             for n in range(N):
    #                 full_size_output[b, s, :, n] = F.interpolate(
    #                     output[b, s, :, n].unsqueeze(0).unsqueeze(0),
    #                     size=original_seq_len,
    #                     mode='linear',
    #                     align_corners=True
    #                 ).squeeze()
        
    #     # 9. 添加残差连接 - 将原始输入与预测输出相加
    #     full_size_output = full_size_output + wavelet_coeffs
        
    #     # 可选：添加门控机制以动态调整残差影响
    #     # gate = torch.sigmoid(self.gate_layer(full_size_output))
    #     # full_size_output = gate * full_size_output + (1-gate) * wavelet_coeffs
        
    #     # 创建注意力权重字典，分开存储环境注意力的两个组件
    #     attention_weights = {
    #         'spatial': spatial_attn,
    #         'temporal': temporal_attn,
    #         'env_self_attention': env_attn_dict['env_self_attention'],
    #         'env_cross_attention': env_attn_dict['cross_attention']
    #     }
        
    #     return full_size_output, attention_weights

    # def forward(self, batch_x, batch_x_mark, decoder_input, batch_y_mark, mask=None, batch_env=None):
    #     """前向传播并返回注意力权重和隐藏特征"""
    #     # 提取小波系数 [B, S, C, N]
    #     wavelet_coeffs = batch_x
        
    #     # 检查环境数据
    #     if batch_env is None:
    #         raise ValueError("环境数据不能为空")
            
    #     # 1. 自适应池化: [B, S, C, N] -> [B, S, pooled_points, N]
    #     B, S, C, N = wavelet_coeffs.shape
    #     original_seq_len = C  # 保存原始序列长度用于上采样
        
    #     # 重组形状以便应用1D池化
    #     # [B, S, C, N] -> [B*S*N, C] -> [B*S*N, 1, C]
    #     reshaped = wavelet_coeffs.permute(0, 1, 3, 2).reshape(B*S*N, C)
    #     reshaped = reshaped.unsqueeze(1)
        
    #     # 应用1D池化
    #     # [B*S*N, 1, C] -> [B*S*N, 1, pooled_points]
    #     pooled = self.adaptive_pool(reshaped)
        
    #     # 恢复原始形状
    #     # [B*S*N, 1, pooled_points] -> [B, S, N, pooled_points] -> [B, S, pooled_points, N]
    #     pooled = pooled.squeeze(1).reshape(B, S, N, self.pooled_points)
    #     pooled = pooled.permute(0, 1, 3, 2)
        
    #     # 重塑为 [B, S*pooled_points, N]
    #     reshaped = pooled.reshape(B, S * self.pooled_points, N)
        
    #     # 2. 空间注意力
    #     spatial_output, spatial_attn = self.spatial_attention(reshaped, return_attention=True)
        
    #     # 3. 时间注意力
    #     temporal_output, temporal_attn = self.temporal_attention(spatial_output, return_attention=True)
        
    #     # 4. 环境注意力
    #     env_output, env_attn_dict = self.env_attention(reshaped, batch_env, return_attention=True)
        
    #     # 5. 融合层
    #     output = self.fusion(spatial_output, temporal_output, env_output)  # [B, S*P, D]
        
    #     # 收集特征和注意力权重
    #     # 注意：空间和时间注意力输出维度为[B, S*P, D]，环境注意力输出维度为[B, S*P, T]
    #     # 为了合并所有特征，我们需要处理不同的维度
    #     hidden_states = {
    #         'spatial': spatial_output,  # [B, S*P, D]
    #         'temporal': temporal_output,  # [B, S*P, D]
    #         'environmental': env_output,  # [B, S*P, T]
    #         'fused': output  # [B, S*P, D]
    #     }
        
    #     # 创建注意力权重字典，分开存储环境注意力的两个组件
    #     attention_weights = {
    #         'spatial': spatial_attn,
    #         'temporal': temporal_attn,
    #         'env_self_attention': env_attn_dict['env_self_attention'],
    #         'env_cross_attention': env_attn_dict['cross_attention']
    #     }
        
    #     final_output = self.final_projection(output)
        
    #     # 将预测输出转换为适合的形状
    #     # [B, S*P, N] -> [B, S, P, N]
    #     output = final_output.reshape(B, S, self.pooled_points, N)
        
    #     # 7. 重塑为便于上采样的格式
    #     # 注意：output已经是[B, S, P, N]形状
        
    #     # 8. 对每个batch、每个scale、每个传感器进行上采样
    #     full_size_output = torch.zeros((B, S, original_seq_len, N), device=output.device)
        
    #     for b in range(B):
    #         for s in range(S):
    #             for n in range(N):
    #                 full_size_output[b, s, :, n] = F.interpolate(
    #                     output[b, s, :, n].unsqueeze(0).unsqueeze(0),
    #                     size=original_seq_len,
    #                     mode='linear',
    #                     align_corners=True
    #                 ).squeeze()
        
    #     # 9. 添加残差连接 - 将原始输入与预测输出相加
    #     full_size_output = full_size_output + wavelet_coeffs
        
    #     return full_size_output, hidden_states, attention_weights
    def forward(self, batch_x, batch_x_mark, decoder_input, batch_y_mark, mask=None, batch_env=None):
            """前向传播并返回注意力权重和隐藏特征"""
            # 提取小波系数 [B, S, C, N]
            wavelet_coeffs = batch_x
            
            # 检查环境数据
            if batch_env is None:
                raise ValueError("环境数据不能为空")
                
            # 1. 自适应池化: [B, S, C, N] -> [B, S, pooled_points, N]
            B, S, C, N = wavelet_coeffs.shape
            original_seq_len = C
            

            # 处理输入的多尺度数据
            batch_x = {
                'cA3': cA3,
                'cD3': cD3,
                'cD2': cD2,
                'cD1': cD1
            }
            # 2. 空间注意力
            spatial_output, spatial_attn = self.spatial_attention(wavelet_coeffs, return_attention=True)
            
            # 3. 时间注意力
            temporal_output, temporal_attn = self.temporal_attention(spatial_output, return_attention=True)
            
            # 4. 环境注意力
            env_output, env_attn_dict = self.env_attention(wavelet_coeffs, batch_env, return_attention=True)
            
            # 5. 融合层
            output = self.fusion(spatial_output, temporal_output, env_output)  # [B, S*P, D]
            
            # 收集特征和注意力权重
            # 注意：空间和时间注意力输出维度为[B, S*P, D]，环境注意力输出维度为[B, S*P, T]
            # 为了合并所有特征，我们需要处理不同的维度
            hidden_states = {
                'spatial': spatial_output,  # [B, S*P, D]
                'temporal': temporal_output,  # [B, S*P, D]
                'environmental': env_output,  # [B, S*P, T]
                'fused': output  # [B, S*P, D]
            }
            
            # 创建注意力权重字典，分开存储环境注意力的两个组件
            attention_weights = {
                'spatial': spatial_attn,
                'temporal': temporal_attn,
                'env_self_attention': env_attn_dict['env_self_attention'],
                'env_cross_attention': env_attn_dict['cross_attention']
            }
            
            final_output = self.final_projection(output)
            
            # 将预测输出转换为适合的形状
            # [B, S*P, N] -> [B, S, P, N]
            output = final_output.reshape(B, S, self.pooled_points, N)
            
            # 7. 重塑为便于上采样的格式
            # 注意：output已经是[B, S, P, N]形状
            
            # 8. 对每个batch、每个scale、每个传感器进行上采样
            full_size_output = torch.zeros((B, S, original_seq_len, N), device=output.device)
            
            for b in range(B):
                for s in range(S):
                    for n in range(N):
                        full_size_output[b, s, :, n] = F.interpolate(
                            output[b, s, :, n].unsqueeze(0).unsqueeze(0),
                            size=original_seq_len,
                            mode='linear',
                            align_corners=True
                        ).squeeze()
            
            # 9. 添加残差连接 - 将原始输入与预测输出相加
            full_size_output = full_size_output + wavelet_coeffs
            
            return full_size_output, hidden_states, attention_weights
    
    @staticmethod
    def get_requirements_for_initialization(variable_name, requirements_dict=None):
        """获取模型初始化所需的参数要求"""
        if requirements_dict is None:
            requirements_dict = {}
        
        # 只添加需要从args中获取的参数
        # 移除了自动确定的参数: num_sensors, num_env_sensors, num_wavelets_scales
        requirements_dict['pooled_points'] = 'int'  # 池化后的点数(P)
        requirements_dict['d_model'] = 'int'  # 注意力隐藏维度(D)
        requirements_dict['n_heads'] = 'int'  # 多头注意力头数(H)
        requirements_dict['dropout'] = 'float'  # 丢弃率
        requirements_dict['e_layers'] = 'int'  # 编码器层数，可选
        requirements_dict['d_layers'] = 'int'  # 解码器层数，可选
        # env_seq_len有默认值，可选参数
            
        return requirements_dict


class Model(nn.Module):
    """模型封装类，用于与框架接口"""
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.model = WaveletBridgeFormer(args)
        
    def forward(self, x, x_mark, decoder_input, y_mark, mask=None, env_data=None):
        return self.model(x, x_mark, decoder_input, y_mark, mask, env_data)
    
    # def forward_with_attention(self, x, x_mark, decoder_input, y_mark, mask=None, env_data=None):
    #     return self.model.forward_with_attention(x, x_mark, decoder_input, y_mark, mask, env_data) 


class AttentionTransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer with attention weights output"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, norm_first=False):
        super().__init__()
        # 自注意力
        self.self_attn = MultiheadAttentionWithWeights(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # 交叉注意力
        self.multihead_attn = MultiheadAttentionWithWeights(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # 其他组件
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                return_attention=True):
        """
        传递输入通过解码器层。
        
        Args:
            tgt: 目标序列，即解码器的输入 (required)
            memory: 编码器的输出 (required)
            tgt_mask: 目标序列的掩码 (optional)
            memory_mask: 记忆序列的掩码 (optional)
            tgt_key_padding_mask: 每批目标键的掩码 (optional)
            memory_key_padding_mask: 每批记忆键的掩码 (optional)
            return_attention: 是否返回注意力权重
            
        Returns:
            Tuple of (output, (self_attn, cross_attn)) if return_attention=True,
            otherwise just output
        """
        x = tgt
        self_attn_weights = None
        cross_attn_weights = None
        
        if self.norm_first:
            # 先规范化
            # 自注意力
            attn_output, self_attn_weights = self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask
            )
            x = x + attn_output
            
            # 交叉注意力
            attn_output, cross_attn_weights = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + attn_output
            
            # 前馈网络
            x = x + self._ff_block(self.norm3(x))
        else:
            # 自注意力
            attn_output, self_attn_weights = self._sa_block(
                x, tgt_mask, tgt_key_padding_mask
            )
            x = self.norm1(x + attn_output)
            
            # 交叉注意力
            attn_output, cross_attn_weights = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask
            )
            x = self.norm2(x + attn_output)
            
            # 前馈网络
            x = self.norm3(x + self._ff_block(x))
        
        if return_attention:
            return x, (self_attn_weights, cross_attn_weights)
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """自注意力块"""
        return self.self_attn(x, x, x, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=True)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        """交叉注意力块"""
        return self.multihead_attn(x, mem, mem, attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)

    def _ff_block(self, x):
        """前馈网络块"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class AttentionTransformerDecoder(nn.Module):
    """TransformerDecoder that can return attention weights"""
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        # 制作多个解码器层
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                return_attention=True):
        """
        Args:
            tgt: 目标序列
            memory: 编码器输出的序列
            tgt_mask: 目标序列掩码
            memory_mask: 记忆序列掩码
            tgt_key_padding_mask: 目标序列填充掩码
            memory_key_padding_mask: 记忆序列填充掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 变换后的序列
            attention_weights: 所有层的注意力权重 (如果return_attention=True)
        """
        output = tgt
        self_attns = []
        cross_attns = []

        for mod in self.layers:
            if return_attention:
                output, (self_attn, cross_attn) = mod(
                    output, memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=True
                )
                self_attns.append(self_attn)
                cross_attns.append(cross_attn)
            else:
                output = mod(
                    output, memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )

        if self.norm is not None:
            output = self.norm(output)

        if return_attention:
            return output, {'self_attention': self_attns, 'cross_attention': cross_attns}
        return output 