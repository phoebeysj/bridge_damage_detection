import pandas as pd
import os
from tools import result_save, percentile_cal, damage_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import sys

parent_path = '/root/autodl-tmp/experiments_minidata/loss_and_prediction/Bridgeformer/sensor3/mse_orsr'
health_path = os.path.join(parent_path, 'test_health/label_loss.csv')
test_path = os.path.join(parent_path, 'test/label_loss.csv')
vali_path = os.path.join(parent_path, 'vali/label_loss.csv')
sys.stdout = open(os.path.join(parent_path,'output.txt'), 'w')

vali_df = pd.read_csv(vali_path)
health_df = pd.read_csv(health_path)
test_df = pd.read_csv(test_path)

vali_df['losses'] = - vali_df['losses']
health_df['losses'] = - health_df['losses']
test_df['losses'] = - test_df['losses']

loss_mean, percentile = percentile_cal(vali_df['losses'])

test_df['pred_label'] = test_df['losses'] > percentile
health_df['pred_label'] = health_df['losses'] > percentile
damage_metric(flag='test', ts = test_df)
damage_metric(flag='test_health', ts = health_df)

df_all = pd.concat([test_df, health_df], ignore_index=True)

y_true = df_all['true_label']
y_pred = df_all['pred_label']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

sys.stdout.close()
    

