import os
from utils.tools import batch_visualize_attention

# 设置要分析的目录
model_name = "Autoformer"
data_name = "bridge_data"

# 分析所有实验结果
base_dir = f"./checkpoints/{model_name}_{data_name}"
for exp_dir in os.listdir(base_dir):
    attn_dir = os.path.join(base_dir, exp_dir, 'attention_weights')
    if os.path.exists(attn_dir):
        print(f"处理实验: {exp_dir}")
        batch_visualize_attention(attn_dir)