# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import keras
from keras.layers import  Input,Dense
from keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

X_train_scaled = pd.read_csv('./scaled_train_X.csv',delimiter=',')
y_train = pd.read_csv('./train_y.csv',delimiter=',')
y_train=y_train.iloc[:,1:]

submission = pd.read_csv('./sample_submission.csv', index_col='seg_id')
X_test_scaled = pd.read_csv('X_test_scaled.csv',delimiter=',',header=None)

X_test_scaled=X_test_scaled.iloc[:,:]
X_train_scaled=X_train_scaled.iloc[:,2:]

dim=X_train_scaled.shape[1]
dim_out=3
epochs=3
bias=7.5
n_fold = 5
fls=51
sls=52

inputs = Input(shape=(fls,))
layer1 = Dense(32, activation='relu',bias=bias)(inputs)
layer2 = Dense(16, activation='relu',bias=bias)(layer1)

inputs2 = Input(shape=(sls,))
layer3 = Dense(32, activation='relu',bias=bias)(inputs2)
layer4 = Dense(16, activation='relu',bias=bias)(layer3)
added=keras.layers.Concatenate(axis=-1)([layer2, layer4])

layer5 = Dense(16, activation='relu',bias=bias)(added)
predictions = Dense(dim_out, activation='relu',bias=bias)(layer5)

folds = KFold(n_splits=n_fold, shuffle=True, random_state=64)

y_pred = np.zeros(shape=(len(X_train_scaled),dim_out))

errors = np.zeros(n_fold)

plt.figure(figsize=(6, 6))
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
index=0


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled,y_train.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    
    X_tr, X_val = X_train_scaled.iloc[trn_idx], X_train_scaled.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
        
    model = Model(inputs=[inputs,inputs2], outputs=predictions)
    model.compile(optimizer="Adam",
          loss='mean_absolute_error',# Call the loss function with the selected layer
          metrics=['accuracy'])
    model.fit([X_tr.iloc[:,0:fls],X_tr.iloc[:,fls:-1]], y_tr.iloc[:,(10-dim_out):10],
          epochs=epochs)
    
    y_ppred = model.predict([X_val.iloc[:,0:fls],X_val.iloc[:,fls:-1]])
    y_pred[val_idx] = y_ppred
    y_val=y_val.values
    
    plt.scatter(y_val[:,-1].flatten(), y_ppred[:,-1])
    
    errors[index]=mean_absolute_error(y_val[:,-1], y_ppred[:,-1])
    print(errors[index])
    index=index+1
plt.show()

score = errors.mean()
print(f'error mean: {score:0.3f}')

model = Model(inputs=[inputs,inputs2], outputs=predictions)
model.compile(optimizer="Adam",
      loss='mean_absolute_error',# Call the loss function with the selected layer
      metrics=['accuracy'])

model.fit([X_train_scaled.iloc[:,0:fls],X_train_scaled.iloc[:,fls:-1]], y_train.iloc[:,(10-dim_out):10],
          epochs=epochs)

y_pred=model.predict([X_train_scaled.iloc[:,0:fls],X_train_scaled.iloc[:,fls:-1]])


submission['time_to_failure'] = model.predict([X_test_scaled.iloc[:,0:fls],X_test_scaled.iloc[:,fls:-1]])
submission.to_csv('submission.csv')


plt.figure(figsize=(6, 6))
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.scatter(y_train.iloc[:,-1], y_pred[:,-1])
plt.show()


score = mean_absolute_error(y_train.iloc[:,-1], y_pred[:,-1])
print(f'Score: {score:0.3f}')