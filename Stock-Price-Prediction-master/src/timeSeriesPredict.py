#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:54:11 2017

@author: dhingratul

Predicts the next day (closing) stock prices for S&P 500 data using LSTM,
and 1D conv layer
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import helper
import time
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# Load Data

seq_len = 12
norm_win = True
epoch = 50
filename_train = 'data/sp500.csv'
filename_test = 'data/Medinet Nasr.csv'
data = ['Suez Cement.csv', 'Medinet Nasr.csv',
        'Oriental Weavers.csv', 'T M G Holding.csv', 'Telecom Egypt.csv']
# X_tr, Y_tr, train_data, train = helper.load_data(
#     filename_train, seq_len, norm_win)
X_tr, Y_tr, X_te, Y_te, w0_test, data = helper.load_data(
    'data/' + data[0], seq_len, norm_win)

print("train Shapes : ", X_tr.shape, Y_tr.shape,
      "\n\ntest shapes : ", X_te.shape, Y_te.shape)

# Model Build0
model = Sequential()
model.add(LSTM(units=50, input_shape=(None, 1), return_sequences=True))
# Adding the output layer
model.add(Dropout(0.02))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(200, return_sequences=False))
model.add(Dropout(0.02))
model.add(Dense(output_dim=1))  # Linear dense layer to aggregate into 1 val
model.add(Activation('linear'))

timer_start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('Model built in: ', time.time()-timer_start)
# Training model
model.fit(X_tr,
          Y_tr,
          batch_size=512,
          nb_epoch=epoch,
          validation_split=0.05
          )
# Predictions
win_size = seq_len
pred_len = seq_len
plot = False
if plot:
    pred = helper.predict_seq_mul(model, X_te, win_size, pred_len)
    helper.plot_mul(pred, Y_te, pred_len)
else:
    pred = helper.predict_pt_pt(model, X_te)
    split_ratio = 0.8
    split = round(split_ratio * data.shape[0])
    # mse_model = mean_squared_error(Y_te, pred)
    # X_te_de = helper.denormalize(X_te.reshape(
    #     (X_te.shape[0], X_te.shape[1])), w0_test[int(split):])

    pred = helper.denormalize(pred, w0_test[int(split):])
    Y_te = helper.denormalize(Y_te, w0_test[int(split):])
    # plt.plot(X_te_de[:12, -1], 'ko', alpha=0.8,
    #          markersize=12, label="X test Price")
    plt.plot(Y_te[-30:], 'b',
             alpha=0.9, markersize=10, label="True Price")
    plt.plot(pred[-30:], 'r', alpha=1,markersize=8, label='predicted price')
	plt.title("Suez Cement")
    plt.legend()
    plt.show()
    # size = 10
    # plt.plot(X_te_de[:size, -1], 'ko', markersize=15,
    #          alpha=0.5, label="X Test prices")
    # plt.plot(range(1, size + 1), Y_te[:size], 'bo',
    #          markersize=12, alpha=0.7, label="True Prices")
    # plt.plot(range(1, size + 1), pred[:size], 'ro',
    #          markersize=8, alpha=1, label="Predicted")
    # plt.show()
    # for i in range(size):
    #     print("\nActual price : " , Y_te[-size  + i] , "\npredict price :\n" , pred[-size + i])

    RMSE = np.sqrt(mean_squared_error(Y_te[-30:], pred[-30:]))
    print("\nRMSE : ", RMSE / np.mean(Y_te))

    # print("MSE of DL model ", mse_model)
    # # Stupid Model
    # y_bar = np.mean(X_te, axis=1)
    # y_bar = np.reshape(y_bar, (y_bar.shape[0]))
    # mse_base = mean_squared_error(Y_te, y_bar)
    # print("MSE of y_bar Model", mse_base)
    # # t-1 Model
    # y_t_1 = X_te[:, -1]
    # y_t_1 = np.reshape(y_t_1, (y_t_1.shape[0]))
    # mse_t_1 = mean_squared_error(Y_te, y_t_1)
    # print("MSE of t-1 Model", mse_t_1)
    # # Comparisons
    # improv = (mse_model - mse_base)/mse_base
    # improv_t_1 = (mse_model - mse_t_1)/mse_t_1
    # print("%ge improvement over naive model", improv)
    # print("%ge improvement over t-1 model", improv_t_1)
    # corr_model = np.corrcoef(Y_te, pred)
    # corr_base = np.corrcoef(Y_te, y_bar)
    # corr_t_1 = np.corrcoef(Y_te, y_t_1)
    # print("Correlation of y_bar \n ", corr_base, "\n t-1 model \n", corr_t_1,
    #       "\n DL model\n", corr_model)
