import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score

df_test = pd.read_csv('test10000_label.csv')
df_pred = pd.read_csv('predict.csv')

y_test = []
y_pred = []

for i in df_test['0']:
    y_test.append(int(i))
for i in df_pred['0']:
    y_pred.append(int(i))

print('Results on test data:\n----------------------')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'MCC: {matthews_corrcoef(y_test, y_pred)}')
print(f'ROC Area: {roc_auc_score(y_test, y_pred)}')
