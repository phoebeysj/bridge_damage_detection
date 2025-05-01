import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib import font_manager

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print('警告：未能设置中文字体，中文可能无法正常显示。', e)

def plot_loss_env_by_month(loss_csv_path, env_parquet_path, env_columns=None):
    if env_columns is None:
        env_columns = ['WS', 'WD', 'H', 'AT', 'TE']

    # 1. 读取数据
    df_loss = pd.read_csv(loss_csv_path)
    df_env = pd.read_parquet(env_parquet_path)

    # 2. 时间戳处理
    df_loss['time'] = pd.to_datetime(df_loss['time'], unit='ns')
    df_loss['date'] = df_loss['time'].dt.floor('D')
    df_env = df_env.reset_index()
    df_env['date'] = pd.to_datetime(df_env['time']).dt.floor('D')

    # 3. 只保留关心的环境变量
    keep_cols = ['time', 'date'] + [col for col in env_columns if col in df_env.columns]
    df_env = df_env[keep_cols]

    # 4. 自动获取所有 loss 列名
    loss_columns = [col for col in df_loss.columns if col.endswith('_losses')]

    # 5. 合并
    df_loss_day = df_loss[['date'] + loss_columns].drop_duplicates('date')
    df_env = df_env.merge(df_loss_day, on='date', how='left')

    # 6. 添加月份信息
    df_env['month'] = df_env['time'].dt.month
    months = sorted(df_env['month'].dropna().unique())
    colors = cm.get_cmap('tab20', len(months))

    # 7. 输出目录
    out_dir = os.path.dirname(loss_csv_path)

    # 8. 双重循环画图
    for loss_col in loss_columns:
        for env_col in env_columns:
            if env_col not in df_env.columns:
                continue
            fig, ax1 = plt.subplots(figsize=(16, 6))
            ax2 = ax1.twinx()
            for i, month in enumerate(months):
                mask = (
                    (df_env['month'] == month)
                    & (~df_env[loss_col].isna())
                    & (~df_env[env_col].isna())
                )
                if mask.sum() == 0:
                    continue
                ax1.plot(df_env.loc[mask, 'time'], df_env.loc[mask, loss_col], 
                         color=colors(i), label=f'Loss {month}月', alpha=0.7, linewidth=2)
                ax2.plot(df_env.loc[mask, 'time'], df_env.loc[mask, env_col], 
                         color=colors(i), linestyle='--', label=f'{env_col} {month}月', alpha=0.5)
            ax1.set_ylabel(loss_col)
            ax2.set_ylabel(env_col)
            ax1.set_xlabel('Time')
            plt.title(f'{loss_col} 和 {env_col} 随时间变化（月分色）')
            # 图例合并
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2)
            # 美化x轴
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=30)
            fig.tight_layout()
            out_path = os.path.join(out_dir, f'{loss_col}_{env_col}_monthcolor.png')
            plt.savefig(out_path)
            plt.close(fig)
            print(f'Saved: {out_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('用法: python plot_loss_env_by_month.py <loss_label.csv路径> <env_data.parquet路径>')
        sys.exit(1)
    loss_csv_path = sys.argv[1]
    env_parquet_path = sys.argv[2]
    plot_loss_env_by_month(loss_csv_path, env_parquet_path) 