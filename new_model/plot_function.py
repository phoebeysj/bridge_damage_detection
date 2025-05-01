import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation_v2 import Exp_Imputation_bridge
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from sklearn.metrics import roc_curve, auc
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.tools import plot_mae_distribution, damage_metric, box, plot_true_pred, heatmap_plot,percentile_cal
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='imputation',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/all_datasets', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETT-small/ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/experiments_v2/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--train_loss_comparation', type=str, default='/root/autodl-tmp/experiments_v2/train_plot/', help='location of vali_data,test_data_loss')
    parser.add_argument('--loss_save', type=str, default='/root/autodl-tmp/experiments_v2/loss_and_prediction/', help='location of vali_data,test_data_loss')
    parser.add_argument('--plot_save',type=str,default='/root/autodl-tmp/experiments_v2/plot_save',help='metric result save folder')
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
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
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
    parser.add_argument('--loss_mode', type = str, default = 'mse', help='mse,freq_mse,combined')
    parser.add_argument('--di_mode', type = str, default = 'mse',help='mse, freq_mse, orsr')
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

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation_bridge
    # elif args.task_name == 'anomaly_detection':
    #     Exp = Exp_Anomaly_Detection
    # elif args.task_name == 'classification':
    #     Exp = Exp_Classification
    # else:
    #     Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}'.format(
                #args.task_name,
                #args.model_id,
                #args.model,
                args.loss_mode,
                args.di_mode,
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
                # args.sensors)
            )
            # txt_path = os.path.join(args.loss_save, setting, "output.txt")
            # os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            # orig_stdout = sys.stdout
            # sys.stdout = open(txt_path, "w")
    
            # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.train(setting)

            # print('>>>>>>>testing health data: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting, flag='test_health')

            # print('>>>>>>>testing damaged data: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting, flag='test')
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    # else:
    #     data_path = "data_final.parquet"
    #     exp = Exp(args)  # set experiments
    #     ii = 0
    #     setting = '{}_{}_{}_{}_{}_exp_times:{}'.format(
    #         # args.task_name,
    #         args.model_id,
    #         args.model,
    #         # args.data,
    #         # args.features,
    #         # args.seq_len,
    #         # args.label_len,
    #         # args.pred_len,
    #         # args.d_model,
    #         # args.n_heads,
    #         # args.e_layers,
    #         # args.d_layers,
    #         # args.d_ff,
    #         # args.expand,
    #         # args.d_conv,
    #         # args.factor,
    #         # args.embed,
    #         # args.distil,
    #         # args.des,
    #         args.sensors, 
    #         data_path, ii)

    #     print('>>>>>>>testing_health: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.test(setting, test=1, flag='test_health')
    #     print('>>>>>>>testing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.test(setting, test=1, flag='test')
    #     if args.gpu_type == 'mps':
    #         torch.backends.mps.empty_cache()
    #     elif args.gpu_type == 'cuda':
    #         torch.cuda.empty_cache()
            
# loss_load
    health_csv_path = os.path.join(args.loss_save, setting, 'test_health', 'label_loss.csv') 
    test_csv_path = os.path.join(args.loss_save, setting, 'test', 'label_loss.csv')
    vali_csv_path = os.path.join(args.loss_save, setting, 'vali', 'label_loss.csv')

    # health_weight_path = os.path.join(args.loss_save, setting, 'test_health', 'moe_weight.csv') 
    # test_weight_path = os.path.join(args.loss_save, setting, 'test', 'moe_weight.csv')
    # vali_weight_path = os.path.join(args.loss_save, setting, 'vali', 'moe_weight.csv')    

    save_path = os.path.join(args.plot_save, setting)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # plot_mae_distribution(test_csv_path, args.sensors,'D1', save_path)
    # plot_mae_distribution(test_csv_path, args.sensors,'D2', save_path)
    # plot_mae_distribution(test_csv_path, args.sensors,'D3', save_path)
    # plot_mae_distribution(test_csv_path, args.sensors,'D4', save_path)
    # print('mae_distribution done!')

    df_health = pd.read_csv(health_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df_vali = pd.read_csv(vali_csv_path)
    columns_to_use = ["time", "pred_label", "losses", "true_label"]
    df_health_left = df_health[columns_to_use]
    df_test_left = df_test[columns_to_use]
    df_total = pd.concat([df_health_left, df_test_left], ignore_index=True)
    # box(df_vali['losses'], df_health['losses'], df_test['losses'], 'vali', 'health', 'damage', save_path)
    # print('box done!')
    # txt_path = os.path.join(args.loss_save, setting, "output.txt")
    # os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    # orig_stdout = sys.stdout
    # sys.stdout = open(txt_path, "w")

    # vali_mean, vali_loss_left= percentile_cal(df_vali['losses'],per_num=5)
    # vali_mean, vali_loss_right = percentile_cal(df_vali['losses'],per_num=90)
    # df_test['pred_label'] =(df_test['losses'] < vali_loss_left ) | (df_test['losses'] > vali_loss_right)
    # df_test.to_csv(test_csv_path)
    # df_neg = df_health
    # df_pos = df_test
    # damage_metric('test_health', df_neg)
    # damage_metric('test', df_pos)
    # df_all = pd.concat([df_neg, df_pos], ignore_index=True)
    # y_true = df_all['true_label']
    # y_pred = df_all['pred_label']
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 Score: ", f1)
    # sys.stdout.close()
    # sys.stdout = orig_stdout
    # print(f'{args.sensors}is done!')

    # moe_weight_vali = pd.read_csv(vali_weight_path)
    # moe_weight_health = pd.read_csv(health_weight_path)
    # moe_weight_test = pd.read_csv(test_weight_path)    
    # heatmap_plot([moe_weight_vali, moe_weight_health, moe_weight_test],['vali','health','test'] ,save_path)
    # health_value = os.path.join(args.loss_save, setting, 'test_health', 'TRUE_PRED_VALUE.csv')
    # test_value = os.path.join(args.loss_save, setting, 'test', 'TRUE_PRED_VALUE.csv')
    # vali_value = os.path.join(args.loss_save, setting, 'vali', 'TRUE_PRED_VALUE.csv')

    # plot_true_pred(health_value, save_path, flag='health')
    # plot_true_pred(test_value, save_path, flag='test')
    # plot_true_pred(vali_value, save_path, flag='vali')
   
    # print('plot done!')
    
    
    
# loss_modes = ['mse', 'freq_mse', 'combined']
# di_modes = ['mse', 'freq', 'orsr']
# metrics = ['Health_Accuracy', 'Damage_Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
# parent_path = '/root/autodl-tmp/experiments_ourmodel/loss_and_prediction/Bridgeformer'
# for sensor in ['sensor6','sensor10','sensor7','sensor12','sensor14','sensor16']:
#     base_path = os.path.join(parent_path, sensor)
#     data_dict = {metric: np.zeros((3, 3)) for metric in metrics}

#     for i, loss_mode in enumerate(loss_modes):
#         for j, di_mode in enumerate(di_modes):
#             file_path = os.path.join(base_path, f"{loss_mode}_{di_mode}", "output.txt")
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 lines = [line.strip() for line in f if line.strip()] 
#                 if len(lines) >= 6:
#                         data_dict['Health_Accuracy'][i][j] = float(lines[0].split('：')[-1])
#                         data_dict['Damage_Accuracy'][i][j] = float(lines[1].split('：')[-1])
#                         data_dict['Accuracy'][i][j] = float(lines[2].split(':')[-1])
#                         data_dict['Precision'][i][j] = float(lines[3].split(':')[-1])
#                         data_dict['Recall'][i][j] = float(lines[4].split(':')[-1])
#                         data_dict['F1_Score'][i][j] = float(lines[5].split(':')[-1])

#     plt.figure(figsize=(15, 10))
#     for idx, metric in enumerate(metrics):
#         plt.subplot(2, 3, idx+1)
#         df = pd.DataFrame(data_dict[metric], index=loss_modes, columns=di_modes)
#         sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', vmin=0.7, vmax=1.0)
#         plt.title(metric)
#         if idx >= 3: plt.xlabel('di_mode')
#         if idx % 3 == 0: plt.ylabel('loss_mode')

#     plt.tight_layout()
#     output_path = os.path.join(base_path, 'heatmap_results.png')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"热力图已保存至: {output_path}")
    print(df_total["true_label"].value_counts())  
    macro_size = 60 
    macro_labels = []
    macro_true = []
    for i in range(0, len(df_total) - macro_size + 1, macro_size):
        window = df_total.iloc[i:i+macro_size]
        pred_damaged_ratio = window['pred_label'].sum() / macro_size
        true_macro_label = int(window['true_label'].mean() > 0)  # 假设只要里面有一个是损伤就算损伤
        macro_labels.append(pred_damaged_ratio)
        macro_true.append(true_macro_label)
    fpr, tpr, thresholds = roc_curve(macro_true, macro_labels)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.style.use('bmh')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Macrosequence)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(save_path, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    plt.figure(figsize=(12, 4))
    plt.plot(macro_labels, label='Damage Probability per Macrosequence')
    plt.axhline(best_threshold, color='red', linestyle='--', label=f'T2 threshold ({best_threshold:.2f})')
    plt.xlabel('Macrosequence Index')
    plt.ylabel('Predicted Damage %')
    plt.title('Control Chart - ORSR-based Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    control_path = os.path.join(save_path, 'control_chart.png')
    plt.savefig(control_path)
    plt.close()

