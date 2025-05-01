from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, inject_sensor_fault, diversity_loss
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


warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import os

# def box(loss,path):
#     plt.figure()
#     plt.hist(loss, bins=10, edgecolor='black')  # bins 参数设置柱子的数量
#     plt.xlabel('Loss 值')
#     plt.ylabel('频数')
#     plt.title('loss straight square figure')
#     plt.savefig(os.path.join(path, 'loss直方图.png'), dpi=300)
#     plt.show()
import os
import matplotlib.pyplot as plt



def box(loss, path, flag:str, second_loss=None):
    """
    如果 second_loss 为 None,就只画一个直方图;
    如果 second_loss 不为 None,就在同一个图上叠加两个直方图.
    """
    os.makedirs(path, exist_ok=True)

    if isinstance(loss, torch.Tensor):
        loss = loss.detach().cpu().numpy()
    if second_loss is not None and isinstance(second_loss, torch.Tensor):
        second_loss = second_loss.detach().cpu().numpy()

    if loss.shape[1] == 1:
        plt.figure(figsize=(8, 6))
        if second_loss is None:
            plt.hist(loss, bins=20000, edgecolor='black')
            plt.xlabel('Loss value')
            plt.ylabel('nums')
            plt.title('Loss square comparation')
            plt.xlim(0, 2)
            save_name = flag + 'loss直方图.png'
        else:
            plt.hist(loss, bins=20000, alpha=0.5, edgecolor='black', label='Loss_vali')
            plt.hist(second_loss, bins=20000, alpha=0.5, edgecolor='black', label='Loss_test')
            plt.xlabel('Loss value')
            plt.ylabel('nums')
            plt.title('Loss square comparation')
            plt.xlim(0, 2)
            plt.legend()
            save_name =  flag + 'loss对比直方图.png'
        plt.savefig(os.path.join(path, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 多传感器情况：loss 是 2D 数组 (num_samples, num_sensors)
    else:
        num_sensors = loss.shape[1]
        fig, axes = plt.subplots(num_sensors, 1, figsize=(8, num_sensors * 3), sharex=True)
        if num_sensors == 1:
            axes = [axes]
        
        for i in range(num_sensors):
            if second_loss is None:
                axes[i].hist(loss[:, i], bins=20000, edgecolor='black')
                axes[i].set_title(f'Sensor {i+1} Loss  square comparation')
            else:
                axes[i].hist(loss[:, i], bins=20000, alpha=0.5, edgecolor='black', label='Loss_1')
                axes[i].hist(second_loss[:, i], bins=20000, alpha=0.5, edgecolor='black', label='Loss_2')
                axes[i].set_title(f'Sensor {i+1} Loss  square comparation')
                axes[i].legend()
            axes[i].set_xlim(0, 2)
            axes[i].set_xlabel('Loss 值')
            axes[i].set_ylabel('频数')
        plt.tight_layout()
        
        save_name =  flag + 'loss直方图.png' if second_loss is None else  flag + 'loss对比直方图.png'
        plt.savefig(os.path.join(path, save_name), dpi=300, bbox_inches='tight')
        plt.close()

    # if second_loss is None:
    #     # 只绘制一个直方图
    #     plt.hist(loss, bins=500, edgecolor='black')
    #     plt.xlabel('Loss 值')
    #     plt.ylabel('频数')
    #     plt.title('vali Loss 直方图')
    #     plt.xlim(0, 0.1)
    #     save_name =  'loss直方图.png'
    # else:
    #     # 叠加两个直方图
    #     plt.hist(loss, bins=500, alpha=0.5, edgecolor='black', label='Loss_vali')
    #     plt.hist(second_loss, bins=500, alpha=0.5, edgecolor='black', label='Loss_test')
    #     plt.xlabel('Loss value')
    #     plt.ylabel('num')
    #     plt.title('Loss comparation')
    #     plt.xlim(0, 0.1)
    #     plt.legend()
    #     save_name =  'loss对比直方图.png'

    # # 保存并显示
    # os.makedirs(path, exist_ok=True)
    # plt.savefig(os.path.join(path, save_name), dpi=300)
    # # plt.show()
    # plt.close()


def plot_loss(train_loss, vali_loss, test_vali_loss, folder, patience, loss_type):
    # 创建保存路径)
    if test_vali_loss != None:
        plot_path = os.path.join(folder, 'tvt_loss comparation')
    else:
        plot_path = folder
    os.makedirs(plot_path, exist_ok=True)  # 如果路径不存在，创建路径
    # train_plot = [loss.cpu().detach() for loss in train_loss]
    # vali_plot = [loss.cpu().detach() for loss in vali_loss]

    # 绘制图形
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(vali_loss, label='Validation Loss')
    if test_vali_loss is not None:
        plt.plot(test_vali_loss, label='Test Loss')

    # 添加标题和标签
    title = 'Train loss, Validation loss'
    if test_vali_loss is not None:
        title += ', Test Loss'
    plt.title(title)
    plt.xlabel('Timestamp')  # 横轴标签为 Timestamp
    plt.ylabel('Loss')
    plt.legend()

    # 保存图像
    filename = f"epoch_loss_{loss_type}_{patience}.png"
    plt.savefig(os.path.join(plot_path, filename), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，以便不会占用过多内存

def percentile_cal(loss):    
    loss_list = [loss_part.cpu().detach() for loss_part in loss]
    loss_tensor = torch.stack(loss_list, dim=0)
    loss_mean = loss_tensor.mean(dim=0).numpy()
    percentile = np.percentile(loss_tensor.numpy(), 90, axis=0)
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


class Bridge_loss(nn.Module):

    def __init__(self, error_type: str = 'mse'):
        super(Bridge_loss, self).__init__()
        if error_type not in ['mse', 'mae', 'rmse']:
            raise ValueError(f"Invalid error_type: {error_type}")
        self.error_type = error_type

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # 假设 pred 和 true 的形状均为 (8193, 2048, 8)
        # 如果使用 torch 操作，下面代码也可以直接用 torch 来实现
        if self.error_type in ['mse', 'rmse']:
            calculation = (pred - true) ** 2
        elif self.error_type == 'mae':
            calculation = torch.abs(pred - true)
        
        # 沿第二维（axis=1）累加误差-> shape: (batch_size, feature_nums)
        error_sum = torch.sum(calculation, dim=1, keepdim=False)
        # 沿第一维（axis=0）求均值 -> shape: (feature_nums)
        error_mean = torch.mean(error_sum, dim=0, keepdim=False)
        
        # 如果是 rmse，需要取平方根
        if self.error_type == 'rmse':
            final_error = torch.sqrt(error_mean)
        else:
            final_error = error_mean
        
        return final_error
    
class Bridge_loss_vali(nn.Module):

    def __init__(self, error_type: str = 'mse'):
        super(Bridge_loss_vali, self).__init__()
        if error_type not in ['mse', 'mae', 'rmse']:
            raise ValueError(f"Invalid error_type: {error_type}")
        self.error_type = error_type

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # 假设 pred 和 true 的形状均为 (8193, 2048, 8)
        # 如果使用 torch 操作，下面代码也可以直接用 torch 来实现
        if self.error_type in ['mse', 'rmse']:
            calculation = (pred - true) ** 2
        elif self.error_type == 'mae':
            calculation = torch.abs(pred - true)
        
        # 沿第二维（axis=1）累加误差，保留维度 -> shape: (8193, 1, 8)
        error_sum = torch.sum(calculation, dim=1, keepdim=False)
        # # 沿第一维（axis=0）求均值 -> shape: (1, 1, 8)
        # error_mean = torch.mean(error_sum, dim=0, keepdim=False)
        
        # 如果是 rmse，需要取平方根
        if self.error_type == 'rmse':
            final_error = torch.sqrt(error_sum)
        else:
            final_error = error_sum
        
        return final_error

    
class Exp_Imputation_bridge(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation_bridge, self).__init__(args)
        self.fault_config = {
            'normal': 0.7,
            'missing': 0.05,
            'minor': 0.05,
            'square': 0.05,
            'trend': 0.05,
            'spike': 0.05,
            'outlier': 0.03,
            'jump': 0.02
        }

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = Bridge_loss('mse')
        return criterion
    
    def select_criterion_vali(self):
        criterion = Bridge_loss_vali('mse')
        return criterion 
    
    def test_memory_activation(self, setting):
        """ 记忆激活测试 """
        path = os.path.join(self.args.checkpoints, setting, 'memory_activation')
        os.makedirs(path, exist_ok=True)
        
        # 生成9种测试模式（正常+8种故障）
        test_patterns = []
        with torch.no_grad():
            for pattern_id in range(9):
                # 生成指定模式数据（需要实现generate_pattern方法）
                data = self.generate_pattern(pattern_id)  
                _, weights = self.model(data)
                
                # 可视化单个模式激活
                plt.figure()
                plt.bar(range(9), weights.mean(dim=0).cpu().numpy())
                plt.title(f"Pattern {pattern_id} Activation")
                plt.savefig(os.path.join(path, f'pattern_{pattern_id}.png'))
                plt.close()
        
        print(f"Memory activation results saved to {path}")

    def online_update(self, new_data_loader, update_steps=50):
        """ 在线更新记忆模块 """
        # 冻结主模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.memory.requires_grad = True
        
        optimizer = optim.Adam([self.model.memory], lr=1e-5)
        
        for step in range(update_steps):
            for batch in new_data_loader:
                # 仅使用正常数据更新
                inputs = batch[0].to(self.device)
                recon, _ = self.model(inputs)
                loss = F.mse_loss(recon, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        print(f"Online updated memory module with {len(new_data_loader.dataset)} samples")

    def detect_anomaly(self, x, alpha=0.7):
        """ 综合异常检测 """
        self.model.eval()
        criterion = self.select_criterion_vali()
        with torch.no_grad():
            recon, weights = self.model(x)

            error = criterion(recon, x).mean(dim=(1,2))
            
            # 记忆激活熵
            entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=1)
            
            # 时空一致性指标
            spatial_var = torch.var(recon - x, dim=2).mean(dim=1)
            temporal_var = torch.var(recon - x, dim=1).mean(dim=1)
            
        return alpha*error + 0.2*entropy + 0.05*(spatial_var + temporal_var)

    def vali(self, vali_data, vali_loader, criterion):
        # path = os.path.join(folder, setting, flag)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                x = x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x = inject_sensor_fault(x,
                                              self.fault_config,
                                              device = self.device
                                              )

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                # inp = batch_x.masked_fill(mask == 0, 0)
                inp = batch_x

                outputs,attn = self.model(inp, batch_x_mark, None, None, mask) ##transformer系列的输出
                # outputs = self.model(batch_x, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                preds.append(pred.reshape(64 * 1000, -1))
                trues.append(true.reshape(64 * 1000, -1))

                if (i + 1) % 20 == 0:
                    print("\titers: {0} | loss: {1:.7f}".format(i + 1, loss.mean().item()))

            preds = torch.cat(preds,dim = 0)
            trues = torch.cat(trues, dim = 0)
            total_loss = torch.cat(total_loss, dim=0)
            # visual(trues, preds, path)
            # print('true_pred_comparation_plot done!')                
        # total_loss_avg = torch.stack(total_loss, dim=0) 
        # total_loss_avg = total_loss_avg.mean(dim=0)  
        # total_loss_std = torch.Tensor(np.std(total_loss,axis=0))
        # total_loss =  torch.Tensor(np.average(total_loss))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        path_train = os.path.join(self.args.train_loss_comparation, setting)
        if not os.path.exists(path_train):
            os.makedirs(path_train)


        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_vali = self.select_criterion_vali()


        epoch_train_mean = []
        epoch_vali_mean = []
        epoch_test_mean = []
        
        epoch_train_std = []
        epoch_vali_std = []
        epoch_test_std = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # if batch_x.sum() == 0:
                #     continue
                iter_count += 1
                model_optim.zero_grad()

                x = x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x = inject_sensor_fault(x,
                                            self.fault_config,
                                            self.device
                                        )
                if i == 1:
                   plot_show(x, batch_x) 
                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs, attn = self.model(batch_x, batch_x_mark, None, None, mask) ##transformer系列的输出
                # outputs = self.model(batch_x, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs, x) # 输入最好是(batch_size, time_steps, sensor_type) 输出就是(sensor_type) #  对比的是原始的x和重构输出的x
                # recon_loss = criterion(outputs, batch_x)
                # diver_loss = diversity_loss(mem.unsqueeze(0),mem.unsqeeze(1).mean())
                # loss = recon_loss + 0.2 * diver_loss
                train_loss.append(loss) 

                if (i + 1) % 20 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.mean().item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # if (i + 1) % 5 == 0:
                #     mem = self.model.memory.detach().cpu().numpy()
                #     plt.figure(figsize=(10,6))
                #     sns.heatmap(mem, cmap='viridis', annot=True)
                #     plt.title(f"Epoch {epoch} Memory Patterns")
                #     plt.savefig(os.path.join(path_train, 'memory',str(i)))
                #     plt.close()

                loss.mean().backward() # 针对多变量的情况,做平均loss的下降
                model_optim.step()
            train_loss = torch.stack(train_loss, dim=0)
            train_loss_mean, train_loss_std = percentile_cal(train_loss)
            epoch_train_mean.append(train_loss_mean)
            epoch_train_std.append(train_loss_std)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            print("start validate and test!")
            vali_loss = self.vali(vali_data, vali_loader, criterion_vali) ## 返回一个列表(sample_length, feature_nums)
            test_loss = self.vali(test_data, test_loader, criterion_vali)
            vali_loss_mean, vali_loss_std = percentile_cal(vali_loss)
            test_loss_mean, test_loss_std = percentile_cal(test_loss)
            epoch_vali_mean.append(vali_loss_mean)
            epoch_vali_std.append(vali_loss_std)
            epoch_test_mean.append(test_loss_mean)
            epoch_test_std.append(test_loss_std)
            # box(vali_loss, path_box_plot, None, 'vali')
            # box(test_loss, path_box_plot, None, 'test')
            # loss_save(vali_loss, path_loss_save,'vali')
            # loss_save(test_loss, path_loss_save,'test')            

            print("Epoch: {0}, Steps: {1} | Train Loss mean : {2} Vali Loss mean: {3}, Vali loss percentile 9 : {4} Test Loss mean: {5} Test loss  percentile 9:{6}".format(
                epoch + 1, train_steps, train_loss_mean, vali_loss_mean, vali_loss_std, test_loss_mean,test_loss_std))
            
            self.vali_loss_mean, self.vali_loss_std = early_stopping(vali_loss_mean, vali_loss_std, self.model, path, vali_loss, test_loss)
            if early_stopping.early_stop:
                ## 画图显示loss下降
                print("Early stopping")
                box(early_stopping.best_vali, path_train , 'vali_test_comparation', early_stopping.best_test)
                print("loss_box_comparation completed!")
                break
            box(early_stopping.best_vali, path_train , 'vali_test_comparation', early_stopping.best_test)
            print("loss_box_comparation completed!")
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        plot_loss(epoch_train_mean, epoch_vali_mean, epoch_test_mean, path_train, self.args.patience, loss_type = 'mean')
        plot_loss(epoch_train_std, epoch_vali_std, epoch_test_std, path_train, self.args.patience, loss_type='std')
        plot_loss(epoch_train_std, epoch_vali_std, None, path_train, self.args.patience, loss_type='mean')
        plot_loss(epoch_train_std, epoch_vali_std, None, path_train, self.args.patience, loss_type='std')
        print("epoch loss plotted!")

        best_model_path = path + '/' + 'checkpoint.pth'
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('train done!')

        return self.model

    def test(self, setting, flag, test=0):
        # 创建loss存储和直方图存储的地址
        path_loss_save = os.path.join(self.args.loss_save, setting, flag) # 分训练、验证、测试
        if not os.path.exists(path_loss_save):
            os.makedirs(path_loss_save)
        # path_box_plot = os.path.join(path_loss_save,'box_plot')
        # if not os.path.exists(path_box_plot):
        #     os.makedirs(path_box_plot)

        test_data, test_loader = self._get_data(flag = flag)
        criterion = self.select_criterion_vali()
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting , 'checkpoint.pth')
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.vali_loss = checkpoint['val_loss']
            self.vali_std = checkpoint['val_std']
            print('loaded successfully!')
        else:
            self.vali_loss = self.vali_loss_mean
            self.vali_std = self.vali_loss_std
        anomaly_threshold = self.vali_std 

        preds = []
        trues = []
        masks = []
        labels = []
        losses = []
        data_stamps = []
        # folder_path = '/root/autodl-tmp/experiments_results/test_results/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs,attn = self.model(inp, batch_x_mark, None, None, mask)
                # outputs = self.model(batch_x, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS 
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                sample_labels = loss > torch.Tensor(anomaly_threshold).to(self.device)  # 异常标签：1，正常标签：0

                labels.append(sample_labels)  # 存储标签
                time_stamp = batch_x_mark[:,0]
                data_stamp = pd.to_datetime(time_stamp.cpu().numpy(), unit='ns')
                masks.append(mask.detach().cpu())
                data_stamps.append(data_stamp)
                preds.append(outputs.reshape(64 * 1000, -1).detach().cpu())
                trues.append(batch_x.reshape(64 * 1000, -1).detach().cpu())
                losses.append(loss.detach())
                # if i % 20 == 0:
                #     filled = true[0, :, -1].copy()
                #     filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                #              pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                #     visual(true[0, :, -1], filled, os.path.join(path_loss_save, str(i) + '.pdf'))
        preds = torch.cat(preds,dim = 0)
        trues = torch.cat(trues, dim = 0)
        losses = torch.cat(losses, dim=0)
        labels = torch.cat(labels, dim = 0)
        data_stamps = np.concatenate([ds.to_numpy() for ds in data_stamps])
        timestamps = pd.Series(data_stamps.astype(int))

        # preds = np.concatenate(preds, 0)
        # trues = np.concatenate(trues, 0)
        # masks = np.concatenate(masks, 0)
        # labels = np.concatenate(labels, 0)
        # data_stamps = np.concatenate(data_stamps, 0)
        # losses = np.concatenate(losses, 0)
        print('test shape:', preds.shape, trues.shape, labels.shape, data_stamps.shape,losses.shape)
        visual(trues, preds, path_loss_save)

        # # result save
        # folder_path = '/root/autodl-tmp/results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        # # print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_imputation.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        total_pred_true = torch.stack([trues,preds],dim=2)
        n, D = preds.shape
        total_pred_true = pd.DataFrame(total_pred_true.view(n, -1))
        total_pred_true.to_csv(os.path.join(path_loss_save, 'TRUE_PRED_VALUE.csv'), index=True)

        total_label_loss = torch.stack([labels, losses],dim=2).cpu()
        n, D = labels.shape
        total_label_loss = pd.DataFrame(total_label_loss.view(n, -1))
        ts = pd.concat([timestamps, total_label_loss], axis=1)
        ts.to_csv(os.path.join(path_loss_save, 'label_loss.csv'), index=True)

        # 损伤识别
        if flag == 'test':
            time_threshold = pd.to_datetime("1998-08-04")
            ts.iloc[:,0] = pd.to_datetime(ts.iloc[:,0], unit='ns')
            df_after = ts[ts.iloc[:, 0] > time_threshold]
            sensor = df_after.iloc[:, 1::2]
            for col_name, col_data in sensor.items():
                accuracy = col_data.mean()  
                print(f"{col_name} 的损伤识别准确率为：{accuracy:.4f}")
        return
