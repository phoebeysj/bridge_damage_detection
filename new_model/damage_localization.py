import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.backends
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation_v2 import Exp_Imputation_bridge
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
# from exp.exp_anomaly_detection import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.tools import plot_mae_distribution, damage_metric, box, plot_true_pred
import sys

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='imputation',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='Z24_mask_0', help='model id')
    parser.add_argument('--model', type=str, required=False, default='FEDformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='Z24', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.parquet', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/experiments_v6/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--train_loss_comparation', type=str, default='/root/autodl-tmp/experiments_v6/train_plot/', help='location of vali_data,test_data_loss')
    parser.add_argument('--loss_save', type=str, default='/root/autodl-tmp/experiments_v6/loss_and_prediction/', help='location of vali_data,test_data_loss')
    parser.add_argument('--plot_save',type=str,default='/root/autodl-tmp/experiments_v6/plot_save',help='metric result save folder')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default= 24*4*4, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24 * 4, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0, help='mask ratio')
    # parser.add_argument('--test_data_type', type = str, default = 'damaged',help='test_data_type')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='learned',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=100,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--AE_hidden_size', type=int, default=256,
                        help='AE_hidden_size')
    parser.add_argument('--AE_code_size', type=int, default=128,
                    help='AE_code_size')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--sensors', type=str, default='sensor14',
                        help='the sensor to be predicted :"sensor5", "sensor6", "sensor7","sensor12","sensor14","sensor16","all"')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            # exp = Exp(args)  # set experiments
            setting = '{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                # args.features,
                # args.seq_len,
                # args.label_len,
                # args.pred_len,
                # args.d_model,
                # args.n_heads,
                # args.e_layers,
                # args.d_layers,
                # args.d_ff,
                # args.expand,
                # args.d_conv,
                # args.factor,
                # args.embed,
                # args.distil,
                # args.des,
            )
            
def merge_label_loss(parent_dir, sensor_list, start_time=None, end_time=None):
    """
    从多个传感器目录中读取 label_loss.csv 并按时间戳横向拼接。
    参数：
    - parent_dir: 包含多个 sensor 目录的父目录
    - sensor_list: 传感器编号列表，例如 [3, 5, 6, 7, 10, 12, 14]
    - start_time, end_time: 字符串或 datetime，用于筛选需要的时间段
    返回值：
    - merged_df: 合并后的 DataFrame，列名如 sensor3_loss, sensor5_loss ...
    """
    model= 'FEDformer'
    dfs = []
    merged_df = pd.DataFrame([])
    for sensor_id in sensor_list:
        sensor_dir = 'imputation_Z24_mask_0_'+model+'_sensor'+str(sensor_id)+'_data.parquet_exp_times:0'
        csv_path = os.path.join(parent_dir, sensor_dir, "test", "label_loss.csv")
        
        if not os.path.exists(csv_path):
            print(f"警告: {csv_path} 不存在，跳过。")
            continue
        
        df = pd.read_csv(csv_path)

        if sensor_id == 3:
            merged_df['time'] = pd.to_datetime(df['time'], unit='ns')
            merged_df.set_index('time', inplace=True)
            merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        df.set_index('time', inplace=True)
        df.rename(columns={'losses': f"sensor{sensor_id}_loss"}, inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        dfs.append(df[f"sensor{sensor_id}_loss"])
    if not dfs:
        print("没有任何有效的传感器数据，返回空 DataFrame。")
        return pd.DataFrame()
    df_loss = pd.concat(dfs, axis = 1)
    # if start_time and end_time:
    #     merged_df = merged_df.loc[start_time:end_time]
    df_loss.index = pd.to_datetime(df_loss.index, unit='ns')
    return df_loss

def plot_average_loss(merged_df, save_path=None):
    """
    根据 averages（各传感器平均loss）绘制柱状图，
    平均loss越大，柱子越深红；平均loss越小，柱子越浅红。
    """
    averages = merged_df.mean()
    # 如果 averages 是空的，就直接返回
    if averages.empty:
        print("averages 为空，无法绘制。")
        return

    min_val = averages.min()
    max_val = averages.max()

    norm = plt.Normalize(vmin=min_val * 0.5, vmax=max_val)
    cmap = plt.cm.Reds  # Reds: 从浅红到深红
    colors = cmap(norm(averages.values))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(averages.index, averages.values, color=colors)
    ax.set_title("Average Loss for Each Sensor")
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Average Loss")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 不需要实际的数据
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Loss Value')  # 颜色条的标签
    
    plt.savefig(save_path)
    print(f"平均 loss 图已保存")
    plt.close()


parent_directory = "/root/autodl-tmp/experiments_v6/loss_and_prediction"
sensor_ids = [3, 5, 6, 7, 12, 14, 16]
start_list = ["1998-08-10", "1998-08-12", "1998-08-17", "1998-08-18", "1998-08-25",  "1998-08-26",  "1998-08-27",  "1998-08-31", "1998-09-02", "1998-09-03","1998-09-06","1998-09-07","1998-09-08"]
end_list = ["1998-08-11", "1998-08-16", "1998-08-18", "1998-08-24", "1998-08-26", "1998-08-27", "1998-08-30", "1998-09-01",  "1998-09-03","1998-09-06", "1998-09-07","1998-09-08","1998-09-09"]

merged_data = merge_label_loss(
    parent_dir=parent_directory, 
    sensor_list=sensor_ids, 
    # start_time=start, 
    # end_time=end
)
print("合并后的 DataFrame:")
print(merged_data.head())
for i in range(len(start_list)):
    start = start_list[i]
    end = end_list[i]
    save_path = os.path.join(args.plot_save, setting, 'damage_detection')
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, start +'_' + end + "loss.png")
    plot_average_loss(merged_data.loc[start:end, :], save_path = save_name)
print('plot done!')
