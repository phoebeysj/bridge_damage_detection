# import subprocess
# import os

# # # sensors = [ "sensor3"]
# # #            #, "sensor5"]
# # sensors =  [ "sensor3", "sensor5", "sensor6","sensor7", "sensor10","sensor12", "sensor14", "sensor16"]
# # loss_modes = ['mse','freq_mse','combined']
# # di_modes = ['mse','freq','orsr']
# # model = "AE"
# # base_cmd = [
# #     "python",
# #     os.path.join(os.environ.get("WORKSPACE_FOLDER", ""), "/root/Time-Series-Library-main/run.py"),
# #     "--task_name", "imputation",
# #     "--is_training", "1",
# #     "--root_path", "/root/autodl-tmp/",
# #     "--data_path", "data.parquet",
# #     "--model_id", "Z24_mask_0",
# #     "--mask_rate", "0",
# #     "--model", model,
# #     "--data", "Z24",
# #     "--features", "M",
# #     "--seq_len", "1000",
# #     "--label_len", "0",
# #     "--pred_len", "0",
# #     "--e_layers", "2",
# #     "--d_layers", "1",
# #     "--factor", "3",
# #     "--enc_in", "1",
# #     "--dec_in", "8",
# #     "--c_out", "1",
# #     "--batch_size", "64",
# #     "--d_model", "128",
# #     "--d_ff", "128",
# #     "--des", "Exp",
# #     "--itr", "1",
# #     "--top_k", "5",
# #     "--learning_rate", "0.001",
# # ]

# # for i in range(len(sensors)):
# #     # for i in range(len(loss_modes)):
# #     #     for j in range(len(di_modes)):
# #             checkpoint_path =os.path.join('/root/autodl-tmp/experiments_ourmodel/checkpoints/', model, sensors[i])
# #             train_loss_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/train_plot/',  model,sensors[i])
# #             loss_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/loss_and_prediction/', model, sensors[i])
# #             plot_save_path =os.path.join('/root/autodl-tmp/experiments_ourmodel/plot_save', model, sensors[i])
# #             os.makedirs(checkpoint_path,exist_ok=True)
# #             os.makedirs(train_loss_path,exist_ok=True)
# #             os.makedirs(loss_save_path,exist_ok=True)
# #             os.makedirs(plot_save_path,exist_ok=True)

# #             # cmd = base_cmd + ["--sensors", sensor] + ["--di_mode", di_modes[i]] + ["--loss_mode", loss_modes[j]]+["--checkpoints", checkpoint_path] + ['--train_loss_comparation',train_loss_path]+['--loss_save',loss_save_path]+['--plot_save', plot_save_path]
# #             cmd = base_cmd + ["--sensors", sensors[i]] + ["--di_mode", 'mse'] + ["--loss_mode", 'mse']+["--checkpoints", checkpoint_path] + ['--train_loss_comparation',train_loss_path]+['--loss_save',loss_save_path]+['--plot_save', plot_save_path]
# #             +['--gpu', i]
# #             print("Running command:", " ".join(cmd))
# #             subprocess.run(cmd) 



# ##  
# import subprocess
# import os
# from multiprocessing import Process

# sensors = ["sensor3"]
# # sensors=["sensor3", "sensor5", "sensor6", "sensor10","sensor7", "sensor12", "sensor14", "sensor16"]
# # loss_modes = ['mse','freq_mse', 'combined']
# # di_modes = ['mse', 'freq','orsr']
# # di_modes = ['orsr']
# # models = ["Bridgeformer",'AE','TimesNet','FEDformer_V1']
# model = "Bridgeformer"
# base_cmd = [
#     "python",
#     os.path.join(os.environ.get("WORKSPACE_FOLDER", ""), "/root/Time-Series-Library-main/run.py"),
#     "--task_name", "imputation",
#     "--is_training", "1",
#     "--root_path", "/root/autodl-tmp/",
#     "--data_path", "data_parquet",
#     "--model_id", "Z24_mask_0",
#     "--mask_rate", "0",
#     "--model", model,
#     "--data", "Z24",
#     "--features", "M",
#     "--seq_len", "1000",
#     "--label_len", "0",
#     "--pred_len", "0",
#     "--e_layers", "2",
#     "--d_layers", "1",
#     "--factor", "3",
#     "--enc_in", "1",
#     "--dec_in", "8",
#     "--c_out", "1",
#     "--batch_size", "64",
#     "--d_model", "128",
#     "--d_ff", "128",
#     "--des", "Exp",
#     "--itr", "1",
#     "--top_k", "5",
#     "--learning_rate", "0.001",
#     "--gpu", "0",
# ]


# # def run_sensor(sensor, gpu):
# #     env = os.environ.copy()
# #     env["CUDA_VISIBLE_DEVICES"] = str(gpu) 

# #     checkpoint_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/checkpoints/', model, sensor)
# #     train_loss_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/train_plot/', model, sensor)
# #     loss_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/loss_and_prediction/', model, sensor)
# #     plot_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/plot_save', model, sensor)
    
# #     os.makedirs(checkpoint_path, exist_ok=True)
# #     os.makedirs(train_loss_path, exist_ok=True)
# #     os.makedirs(loss_save_path, exist_ok=True)
# #     os.makedirs(plot_save_path, exist_ok=True)

# #     for i in range(len(loss_modes)): 
# #         # if i == 0:
# #         #     num_start = 2
# #         # else:
# #         #     num_start = 0
# #         for j in range(len(di_modes)):
# #             cmd = base_cmd + [
# #                 "--sensors", sensor,
# #                 "--di_mode", di_modes[j],
# #                 "--loss_mode", loss_modes[i],
# #                 "--checkpoints", checkpoint_path,
# #                 "--train_loss_comparation", train_loss_path,
# #                 "--loss_save", loss_save_path,
# #                 "--plot_save", plot_save_path
# #             ]
# #             subprocess.run(cmd, env=env)
# def run_sensor(sensor, gpu):
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu) 
#     # sensor = "sensor3"
#     checkpoint_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/checkpoints/', model, sensor)
#     train_loss_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/train_plot/', model, sensor)
#     loss_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/loss_and_prediction/', model, sensor)
#     plot_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/plot_save', model, sensor)
    
#     os.makedirs(checkpoint_path, exist_ok=True)
#     os.makedirs(train_loss_path, exist_ok=True)
#     os.makedirs(loss_save_path, exist_ok=True)
#     os.makedirs(plot_save_path, exist_ok=True)
#     # for i in range(len(loss_modes)): 
#     #     # if i == 0:
#     #     #     num_start = 2
#     #     # else:
#     #     #     num_start = 0
#     #     for j in range(len(di_modes)):
#     cmd = base_cmd + [
#         "--sensors", sensor,
#         "--di_mode",'mse',
#         "--loss_mode", 'mse',
#         "--checkpoints", checkpoint_path,
#         "--train_loss_comparation", train_loss_path,
#         "--loss_save", loss_save_path,
#         "--plot_save", plot_save_path
#     ]
#     subprocess.run(cmd, env=env)

# if __name__ == "__main__":
#     processes = []
#     for idx, sensor in enumerate(sensors):
#         gpu = idx  
#         p = Process(target=run_sensor, args=(sensor, gpu))
#         processes.append(p)
#         p.start()
#     for p in processes:
#         p.join()

#     print("All sensor tasks completed!")


# # import subprocess
# # import os
# # from multiprocessing import Process

# # #sensors = ["sensor6", "sensor10",]
# # sensors=["sensor3", "sensor6", "sensor10","sensor5", "sensor7", "sensor12", "sensor14", "sensor16"]
# # loss_modes = ['mse','freq_mse', 'combined']
# # di_modes = ['mse', 'freq','orsr']
# # # di_modes = ['orsr']
# # models = ["Bridgeformer",'AE','TimesNet','FEDformer_V1']
# # model = "Bridgeformer"
# # base_cmd = [
# #     "python",
# #     os.path.join(os.environ.get("WORKSPACE_FOLDER", ""), "/root/Time-Series-Library-main/plot_function.py"),
# #     "--task_name", "imputation",
# #     "--is_training", "1",
# #     "--root_path", "/root/autodl-tmp/",
# #     "--data_path", "data.parquet",
# #     "--model_id", "Z24_mask_0",
# #     "--mask_rate", "0",
# #     "--model", model,
# #     "--data", "Z24",
# #     "--features", "M",
# #     "--seq_len", "1000",
# #     "--label_len", "0",
# #     "--pred_len", "0",
# #     "--e_layers", "2",
# #     "--d_layers", "1",
# #     "--factor", "3",
# #     "--enc_in", "1",
# #     "--dec_in", "8",
# #     "--c_out", "1",
# #     "--batch_size", "64",
# #     "--d_model", "128",
# #     "--d_ff", "128",
# #     "--des", "Exp",
# #     "--itr", "1",
# #     "--top_k", "5",
# #     "--learning_rate", "0.001",
# #     "--gpu", "0",
# # ]

# # def run_sensor(sensor, gpu):
# #     env = os.environ.copy()
# #     env["CUDA_VISIBLE_DEVICES"] = str(gpu) 
# #     # sensor = "sensor3"
# #     checkpoint_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/checkpoints/', model, sensor)
# #     train_loss_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/train_plot/', model, sensor)
# #     loss_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/loss_and_prediction/', model, sensor)
# #     plot_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/plot_save', model, sensor)
    
# #     os.makedirs(checkpoint_path, exist_ok=True)
# #     os.makedirs(train_loss_path, exist_ok=True)
# #     os.makedirs(loss_save_path, exist_ok=True)
# #     os.makedirs(plot_save_path, exist_ok=True)
# #     for i in range(len(loss_modes)): 
# #         # if i == 0:
# #         #     num_start = 2
# #         # else:
# #         #     num_start = 0
# #         for j in range(len(di_modes)):
# #             cmd = base_cmd + [
# #                 "--sensors", sensor,
# #                 "--di_mode",di_modes[j],
# #                 "--loss_mode",loss_modes[i],
# #                 "--checkpoints", checkpoint_path,
# #                 "--train_loss_comparation", train_loss_path,
# #                 "--loss_save", loss_save_path,
# #                 "--plot_save", plot_save_path
# #             ]
# #             subprocess.run(cmd, env=env)
# # if __name__ == "__main__":
# #     processes = []
# #     for idx, sensor in enumerate(sensors):
# #         gpu = idx  
# #         p = Process(target=run_sensor, args=(sensor, gpu))
# #         processes.append(p)
# #         p.start()
# #     for p in processes:
# #         p.join()

# #     print("All sensor tasks completed!")



# # import subprocess
# # import os
# # from multiprocessing import Process

# # #sensors = ["sensor6", "sensor10",]
# # sensors=["sensor3", "sensor6", "sensor10","sensor5", "sensor7", "sensor12", "sensor14", "sensor16"]
# # loss_modes = ['mse','freq_mse', 'combined']
# # # di_modes = ['mse', 'freq','orsr']
# # di_modes = ['orsr']
# # models = ["Bridgeformer",'AE','TimesNet','FEDformer_V1']
# # model = "Bridgeformer"
# # base_cmd = [
# #     "python",
# #     os.path.join(os.environ.get("WORKSPACE_FOLDER", ""), "/root/Time-Series-Library-main/plot_function.py"),
# #     "--task_name", "imputation",
# #     "--is_training", "1",
# #     "--root_path", "/root/autodl-tmp/",
# #     "--data_path", "data.parquet",
# #     "--model_id", "Z24_mask_0",
# #     "--mask_rate", "0",
# #     "--model", model,
# #     "--data", "Z24",
# #     "--features", "M",
# #     "--seq_len", "1000",
# #     "--label_len", "0",
# #     "--pred_len", "0",
# #     "--e_layers", "2",
# #     "--d_layers", "1",
# #     "--factor", "3",
# #     "--enc_in", "1",
# #     "--dec_in", "8",
# #     "--c_out", "1",
# #     "--batch_size", "64",
# #     "--d_model", "128",
# #     "--d_ff", "128",
# #     "--des", "Exp",
# #     "--itr", "1",
# #     "--top_k", "5",
# #     "--learning_rate", "0.001",
# #     "--gpu", "0",
# # ]

# # def run_sensor(sensor, gpu):
# #     env = os.environ.copy()
# #     env["CUDA_VISIBLE_DEVICES"] = str(gpu) 
# #     # sensor = "sensor3"
# #     checkpoint_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/checkpoints/', model, sensor)
# #     train_loss_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/train_plot/', model, sensor)
# #     loss_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/loss_and_prediction/', model, sensor)
# #     plot_save_path = os.path.join('/root/autodl-tmp/experiments_ourmodel/plot_save', model, sensor)
    
# #     os.makedirs(checkpoint_path, exist_ok=True)
# #     os.makedirs(train_loss_path, exist_ok=True)
# #     os.makedirs(loss_save_path, exist_ok=True)
# #     os.makedirs(plot_save_path, exist_ok=True)
# #     # for i in range(len(loss_modes)): 
# #     #     # if i == 0:
# #     #     #     num_start = 2
# #     #     # else:
# #     #     #     num_start = 0
# #     #     for j in range(len(di_modes)):
# #     cmd = base_cmd + [
# #         "--sensors", sensor,
# #         "--di_mode",'mse',
# #         "--loss_mode", 'mse',
# #         "--checkpoints", checkpoint_path,
# #         "--train_loss_comparation", train_loss_path,
# #         "--loss_save", loss_save_path,
# #         "--plot_save", plot_save_path
# #     ]
# #     subprocess.run(cmd, env=env)
# # if __name__ == "__main__":
# #     processes = []
# #     for idx, sensor in enumerate(sensors):
# #         gpu = idx  
# #         p = Process(target=run_sensor, args=(sensor, gpu))
# #         processes.append(p)
# #         p.start()
# #     for p in processes:
# #         p.join()

# #     print("All sensor tasks completed!")



import subprocess
import os

# # 定义所有需要处理的传感器
# sensors = ["sensor3", "sensor5", "sensor6", "sensor10", "sensor7", "sensor12", "sensor14", "sensor16"]
# # 将所有传感器合并为一个字符串参数
# all_sensors = ",".join(sensors)

model = "Bridgeformer"
checkpoint_path = os.path.join('/root/autodl-tmp/experiments_seperate_without_sample/checkpoints/', model, "all_sensors")
train_loss_path = os.path.join('/root/autodl-tmp/experiments_seperate_without_sample/train_plot/', model, "all_sensors")
loss_save_path = os.path.join('/root/autodl-tmp/experiments_seperate_without_sample/loss_and_prediction/', model, "all_sensors")
plot_save_path = os.path.join('/root/autodl-tmp/experiments_seperate_without_sample/plot_save', model, "all_sensors")

# 创建必要的目录
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(train_loss_path, exist_ok=True)
os.makedirs(loss_save_path, exist_ok=True)
os.makedirs(plot_save_path, exist_ok=True)

# 构建一个使用所有GPU的命令
cmd = [
    "python",
    os.path.join(os.environ.get("WORKSPACE_FOLDER", ""), "/root/new_model/run.py"),
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", "/root/autodl-tmp/",
    "--data_path", "data_parquet",
    "--model_id", "Z24_mask_0",
    "--mask_rate", "0",
    "--model", model,
    "--data", "Z24",
    "--features", "M",
    "--seq_len", "1000",
    "--label_len", "0",
    "--num_sensors",'6',
    "--pred_len", "0",
    "--e_layers", "1",
    "--d_layers", "1",
    "--factor", "3",
    "--enc_in", "8",
    "--dec_in", "8",
    "--c_out", "8",
    "--batch_size", "224",
    "--d_model", "16",
    "--d_ff", "128",
    "--pooled_points", "2048",
    "--n_scales", "3",
    "--env_seq_len", "4",
    "--num_wavelets_scales", "3",
    "--hours_needed", "1",
    "--des", "Exp",
    "--itr", "1",
    "--top_k", "5",
    "--learning_rate", "0.005",
    "--use_multi_gpu",
    "--devices", "0,1,2,3,4,5,6",
    "--sensors", "5,6,7,12,14,16",
    "--di_mode", "orsr",
    "--loss_mode", "mse",
    "--checkpoints", checkpoint_path,
    "--train_loss_comparation", train_loss_path,
    "--loss_save", loss_save_path,
    "--plot_save", plot_save_path
]

# # 将所有命令参数转换为字符串
# cmd = [str(item) for item in cmd]

print("Running command:", " ".join(cmd))
subprocess.run(cmd) 
