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
                 activation="relu", batch_first=False, norm_first=True):
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
    """空间注意力模块：捕捉传感器间的空间关系
    输入: [B, S, C, N]
    输出: [B, S, C, N, D]
    """
    def __init__(self, num_sensors, d_model, nhead, dropout=0.1, e_layers=1):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        encoder_layer = AttentionTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = AttentionTransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=True):
        # x: [B, S, C, N]
        B, S, C, N = x.shape
        # 1. 升维
        x_proj = self.input_projection(x.unsqueeze(-1))  # [B, S, C, N, D]
        # 2. reshape for MHA over sensors
        x_flat = x_proj.reshape(B*S*C, N, -1)  # [B*S*C, N, D]
        residual = x_flat
        # 3. 空间注意力
        if return_attention:
            output, attention_weights = self.transformer_encoder(x_flat, return_attention=True)
        else:
            output = self.transformer_encoder(x_flat)
            attention_weights = None
        # 4. 残差+归一化
        output = self.norm(output + residual)
        # 5. reshape回原始
        output = output.reshape(B, S, C, N, -1)
        if return_attention:
            return output, attention_weights
        return output


class TemporalAttention(nn.Module):
    """时间注意力模块：捕捉时间序列依赖"""
    def __init__(self, num_sensors, d_model, nhead, dropout=0.1, e_layers=1, n_scales=1, pooled_points=384):
        super(TemporalAttention, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
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
        # x: [B, S, C, N, D]
        B, S, C, N, D = x.shape
        # 1. permute/reshape for MHA over coeffs
        x_perm = x.permute(0, 1, 3, 2, 4)  # [B, S, N, C, D]
        x_flat = x_perm.reshape(B*S*N, C, D)  # [B*S*N, C, D]
        # 2. input projection
        x_proj = self.input_projection(x_flat)
        # 3. 残差保存
        residual = x_proj
        # 4. 位置编码
        x_enc = self.pos_encoder(x_proj)
        # 5. MHA
        if return_attention:
            output, attention_weights = self.transformer_encoder(
                x_enc, return_attention=True
            )
        else:
            output = self.transformer_encoder(x_enc)
            attention_weights = None
        # 6. 残差连接+归一化
        output = self.norm(output + residual)
        # 7. reshape回原始
        output = output.reshape(B, S, N, C, D).permute(0, 1, 3, 2, 4)  # [B, S, C, N, D]
        if return_attention:
            return output, attention_weights
        else:
            return output


class EnvironmentalAttention(nn.Module):
    """环境注意力模块：融合环境数据和振动数据"""
    def __init__(self, num_sensors, num_env_sensors, d_model, seq_len, nhead, dropout=0.1, d_layers=1, n_scales=1, pooled_points=384):
        super(EnvironmentalAttention, self).__init__()
        self.n_scales = n_scales
        self.pooled_points = pooled_points
        self.d_model = d_model
        # 振动数据投影
        self.vib_projection = nn.Linear(1, self.d_model)
        
        # 环境数据投影
        self.env_projection = nn.Linear(1, self.d_model)
        self.env_pos_encoder = PositionalEncoding(self.d_model)
        
        # 环境数据自注意力
        env_encoder_layer = AttentionTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
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
        # vib_data: [B, S, C, N]
        # env_data: [B, env_seq_len, num_env_sensors]
        B, S, C, N = vib_data.shape
        _, env_seq_len, num_env = env_data.shape

        # 1. 投影环境数据
        env_proj = self.env_projection(env_data.unsqueeze(-1))  # [B, env_seq_len, num_env, D]
        env_proj = env_proj.reshape(B, env_seq_len * num_env, self.d_model)  # [B, M, D]
        env_proj = self.env_pos_encoder(env_proj)
        env_attended, _ = self.env_transformer(env_proj, return_attention=True)  # [B, M, D]

        # 2. 对每个scale做交叉注意力
        cross_outputs = []
        cross_attns = []
        for s in range(S):
            vib_proj = self.vib_projection(vib_data[:, s, :, :].unsqueeze(-1))  # [B, C, N, D]
            vib_proj = vib_proj.reshape(B, C*N, self.d_model)  # [B, L, D]
            out, attn = self.cross_transformer(vib_proj, env_attended, return_attention=True)
            cross_outputs.append(out.unsqueeze(1))  # [B, 1, L, D]
            cross_attns.append(attn['cross_attention'])
        cross_output = torch.cat(cross_outputs, dim=1)  # [B, S, L, D]

        # 3. reshape回原始振动数据shape [B, S, C, N, D]
        cross_output = cross_output.reshape(B, S, C, N, self.d_model)

        attn_weights = {
            'cross_attention': cross_attns
        }
        return self.norm(cross_output), attn_weights


class FusionLayer(nn.Module):
    """融合层：只在d_model维度融合，然后降维回num_sensors"""
    def __init__(self, d_model, num_sensors):
        super(FusionLayer, self).__init__()
        # 融合后降回d_model
        self.fusion = nn.Linear(d_model * 2,d_model)
        # 降回1维 
        self.output_projection = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spatial_output, temporal_output, env_output=None):
        # 输入: [B, S, C, N, D]
        # 1. 拼接d_model维度
        # 先确保输入shape一致
        # 拼接最后一维d_model
        if env_output is not None:
            fused = torch.cat([spatial_output, temporal_output, env_output], dim=-1)  # [B, S, C, N, D*3]
        else:
            fused = torch.cat([spatial_output, temporal_output], dim=-1)  # [B, S, C, N, D*2]
        # 2. 融合降回d_model
        fused = self.fusion(fused)  # [B, S, C, N, D]
        fused = self.norm(fused)
        # 3. 再降回num_sensors
        output = self.output_projection(fused)  # [B, S, C, N, num_sensors]
        return output


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


class WaveletConvDownsample(nn.Module):
    def __init__(self, num_sensors, out_len=512, chunk_size=30000):
        super().__init__()
        self.out_len = out_len
        self.chunk_size = chunk_size
        self.convs = nn.ModuleDict({
            k: nn.Conv1d(num_sensors, num_sensors, kernel_size=5, padding=2)
            for k in ['cA3', 'cD3', 'cD2', 'cD1']
        })
    
    def forward(self, coeff_dict):
        # 输入: dict of [B, T, N]
        # 输出: [B, S=4, P=512, N]
        out_list = []
        for k in ['cA3', 'cD3', 'cD2', 'cD1']:
            v = coeff_dict[k]  # [B, T, N]
            v = v.permute(0, 2, 1)  # [B, N, T]
            B, N, T = v.shape
            pooled_chunks = []
            for start in range(0, T, self.chunk_size):
                end = min(start + self.chunk_size, T)
                v_chunk = v[:, :, start:end]  # [B, N, chunk]
                v_conv = self.convs[k](v_chunk)  # [B, N, chunk]
                chunk_out_len = max(1, int(self.out_len * (end - start) / T))
                v_down = F.adaptive_max_pool1d(v_conv, chunk_out_len)  # [B, N, chunk_out_len]
                pooled_chunks.append(v_down)
            v_down = torch.cat(pooled_chunks, dim=2)  # [B, N, ~512]
            # 如果拼接后长度和目标out_len不一致，再整体池化一次
            if v_down.shape[2] != self.out_len:
                v_down = F.adaptive_avg_pool1d(v_down, self.out_len)
            v_down = v_down.permute(0, 2, 1)  # [B, 512, N]
            out_list.append(v_down)
        out = torch.stack(out_list, dim=1)  # [B, 4, 512, N]
        return out


class DynamicWaveletUpsample(nn.Module):
    def __init__(self, num_sensors):
        super().__init__()
        # 为每个小波分量创建一个Conv1d，但Upsample的size将在forward时动态确定
        self.conv_refine = nn.ModuleDict({
            k: nn.Conv1d(num_sensors, num_sensors, kernel_size=3, padding=1)
            for k in ['cA3', 'cD3', 'cD2', 'cD1']
        })
    
    def forward(self, x_dict, original_lengths):
        """
        Args:
            x_dict: 字典，包含降采样后的各个分量 {'cA3': [B, pooled, N], ...}
            original_lengths: 字典，包含原始长度 {'cA3': L1, 'cD3': L2, ...}
        Returns:
            dict: 上采样后的各个分量 {'cA3': [B, L1, N], ...}
        """
        out_dict = {}
        for k, x in x_dict.items():
            target_len = original_lengths[k]
            # [B, pooled, N] → [B, N, pooled]
            x = x.permute(0, 2, 1)
            # 动态创建上采样层
            upsample = nn.Upsample(size=target_len, mode='linear', align_corners=True).to(x.device)
            # 上采样
            x_up = upsample(x)  # [B, N, target_len]
            # 可学习微调
            x_refined = self.conv_refine[k](x_up)  # [B, N, target_len]
            # [B, N, target_len] → [B, target_len, N]
            x_refined = x_refined.permute(0, 2, 1)
            out_dict[k] = x_refined
        return out_dict


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

        # 初始化注意力模块
        self.spatial_attention = SpatialAttention(
            self.num_sensors, self.d_model, self.nhead, self.dropout, self.e_layers
        )
        self.temporal_attention = TemporalAttention(
            self.num_sensors, self.d_model, self.nhead, self.dropout, self.e_layers,
            n_scales=self.num_wavelets_scales, pooled_points=self.pooled_points
        )
        # 暂时注释掉环境注意力模块
        self.env_attention = EnvironmentalAttention(
            self.num_sensors, self.num_env_sensors, self.d_model, 
            self.env_seq_len, self.nhead, self.dropout, self.d_layers,
            n_scales=self.num_wavelets_scales, pooled_points=self.pooled_points
        )
        
        # 初始化融合层
        self.fusion = FusionLayer(self.d_model, 1)  # 使用d_model作为输出维度
        
        # # 可选：添加最终输出投影层从d_model映射回num_sensors
        # self.final_projection = nn.Linear(self.d_model, self.num_sensors)
        
        # 不再需要initialized标志
        print(f"模型初始化完成 - 传感器: {self.num_sensors}, 环境传感器: {self.num_env_sensors}, 小波尺度: {self.num_wavelets_scales}")

        # 添加wavelet_conv_downsample
        self.wavelet_conv_downsample = WaveletConvDownsample(num_sensors=self.num_sensors, out_len=self.pooled_points)

        # 添加动态上采样模块
        self.upsampler = DynamicWaveletUpsample(num_sensors=self.num_sensors)

    def forward(self, batch_x, batch_x_mark, decoder_input, batch_y_mark, mask=None, batch_env=None):
        """前向传播"""
        # 保存原始长度
        if isinstance(batch_x, dict):
            original_lengths = {k: v.shape[1] for k, v in batch_x.items()}
            orig_coeff_dict = batch_x  # 保留原始输入
            # 先卷积降采样
            batch_x = self.wavelet_conv_downsample(batch_x)  # [B, 4, 512, N]
        else:
            raise ValueError("batch_x 必须是字典格式，包含不同尺度的小波系数")

        # # 检查环境数据
        # if batch_env is None:
        #     raise ValueError("环境数据不能为空")
        # B, S, C, N = batch_x.shape

        # 2. 空间注意力
        spatial_output, spatial_attn = self.spatial_attention(batch_x, return_attention=True)
        # 3. 时间注意力
        temporal_output, temporal_attn = self.temporal_attention(spatial_output, return_attention=True)
        # 4. 环境注意力 - 暂时注释
        # env_output, env_attn_dict = self.env_attention(batch_x, batch_env, return_attention=True)
        # 临时使用零张量替代环境注意力输出
        # env_output = torch.zeros_like(spatial_output)
        # env_attn_dict = {'cross_attention': None}
        
        # 5. 融合层
        # output = self.fusion(spatial_output, temporal_output, env_output)  # [B, S*P, D]
        output = self.fusion(spatial_output, temporal_output)  # [B, S*P, D]
        # 收集特征和注意力权重
        hidden_states = {
            # 'spatial': spatial_output,  # [B, S*P, D]
            # 'temporal': temporal_output,  # [B, S*P, D]
            # 'environmental': env_output,  # [B, S*P, T]
            'fused': output  # [B, S*P, D]
        }
        attention_weights = {
            'spatial': spatial_attn,
            'temporal': temporal_attn,
            # 'env_self_attention': env_attn_dict['env_self_attention'],
            # 'env_cross_attention': env_attn_dict['cross_attention']
        }
        
        # [B, S*P, N] -> [B, S, P, N]
        output = output.squeeze(-1)
        # 7. 重塑为便于上采样的格式
        # 8. 对每个batch、每个scale、每个传感器进行上采样
        # 直接使用 DynamicWaveletUpsample
        # 准备用于上采样的字典
        pooled_dict = {
            k: output[:, i, :, :].contiguous()  # [B, P, N]
            for i, k in enumerate(['cA3', 'cD3', 'cD2', 'cD1'])
        }

        # 上采样还原,每个尺度恢复到其原始长度
        restored_dict = self.upsampler(pooled_dict, original_lengths)

        # 残差连接：上采样结果 + 原始输入
        final_output = {}
        for k, v in restored_dict.items():
            orig = orig_coeff_dict[k]  # [B, L, N]
            # 对齐长度
            if v.shape == orig.shape:
                final_output[k] = (v + orig).unsqueeze(1)
            else:
                min_len = min(v.shape[1], orig.shape[1])
                final_output[k] = (v[:, :min_len, :] + orig[:, :min_len, :]).unsqueeze(1)

        return restored_dict, hidden_states, attention_weights
    
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




def align_and_stack_wavelet_coeffs(coeff_dict):
    """
    对齐并堆叠小波分量张量
    Args:
        coeff_dict: dict, 例如 {'cA3': [B, T1, N], 'cD3': [B, T2, N], ...}
    Returns:
        tensor: [B, max_T, N, 4]  # 4为分量数
    """
    # 1. 找到最大长度
    lengths = [v.shape[1] for v in coeff_dict.values()]
    max_len = max(lengths)
    batch_size, num_sensors = next(iter(coeff_dict.values())).shape[0], next(iter(coeff_dict.values())).shape[2]
    out_list = []
    for k in ['cA3', 'cD3', 'cD2', 'cD1']:
        v = coeff_dict[k]  # [B, T, N]
        repeat_factor = max_len // v.shape[1]
        remainder = max_len % v.shape[1]
        # 先repeat
        v_rep = v.repeat_interleave(repeat_factor, dim=1)
        # 补齐remainder
        if remainder > 0:
            v_rep = torch.cat([v_rep, v[:, :remainder, :]], dim=1)
        out_list.append(v_rep.unsqueeze(-1))  # [B, max_T, N, 1]
    # 拼接
    out = torch.cat(out_list, dim=-1)  # [B, max_T, N, 4]
    return out 