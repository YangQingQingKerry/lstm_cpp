#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.log(tf.logging.INFO, "TensorFlow version"+tf.__version__)




def fetch_raw(file, symbol):
    path="../data/nyse/{}.csv".format(file)
    df=pd.read_csv(path)
    if (file=="prices") or (file=="prices-split-adjusted"):
        return df[df.symbol==symbol].copy()
    else:
        return df.copy()




def build_timeseries(df, y_col_index, timesteps):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - timesteps
    dim_0 = df.shape[0] - timesteps
    dim_1 = df.shape[1]
    x = np.zeros((dim_0, timesteps, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = df[i:timesteps+i]
        y[i] = df[timesteps+i, y_col_index]
    print("length of time-series i/o: {} {}".format(x.shape,y.shape))
    return x, y


def trim_dataset(df, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = df.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return df[:-no_of_rows_drop]
    else:
        return df




def _create_one_cell(layer, config):
    cell=tf.contrib.rnn.LSTMCell(config.n_neurons[layer], state_is_tuple=True)
    if config.keep_prob == 1.0:
        return cell
    else:
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)


# function to get the next batch
def get_next_batch(perm_array, index_in_epoch, x_t, y_t, batch_size):
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_t.shape[0]:
        start = 0 # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_t[perm_array[start:end]], y_t[perm_array[start:end]], index_in_epoch




