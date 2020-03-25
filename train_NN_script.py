#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:42:58 2020

@author: Dr. Shantanu Shahane
"""

import numpy as np
from matplotlib import pyplot as plt
import timeit
from general_functions import *
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
tic();

folder=''
file='Raw Data.xlsx';
train_fraction=0.8; test_fraction=0.5*(1-train_fraction);
surface_type='SHP'; #options: 'SHL', 'SHP', 'R'
separation=8; #options: 2, 4, 6, 8
surface_temperature=-5; #options: -5, -10, -15

time_column_header=get_time_column_header(surface_type,separation,surface_temperature)

if not os.path.exists(folder+time_column_header):
    os.makedirs(folder+time_column_header)
raw_data=pd.read_excel(folder+file,skiprows=1); column_names=list(raw_data.columns)
time_minutes,frost_mm,train_input,train_output,test_input,test_output,val_input,val_output=get_data(raw_data, time_column_header, train_fraction, test_fraction, False);
keras_model_file=folder+time_column_header+'/keras.h5'
keras_model_epoch_file=folder+time_column_header+'/keras epoch.csv'
#del raw_data

if os.path.isfile(keras_model_file): #already trained model exists
    model=keras.models.load_model(keras_model_file); model.summary();
else: #model is not yet trained; thus define it, train and save
    l2_reg_lambda=0.00005; learning_rate=0.002; EPOCHS=100; dropout_factor=0.0;
    n_hidden_layers=10; n_hidden_units=100;
    write_NN_hyperparameters(folder,time_column_header, l2_reg_lambda, learning_rate, EPOCHS, dropout_factor, n_hidden_layers, n_hidden_units)
    n_hidden_units=n_hidden_units*np.ones((n_hidden_layers,));
    model = build_NN_model(1,1,n_hidden_layers,n_hidden_units,l2_reg_lambda,learning_rate,dropout_factor); model.summary()
    csv_logger=keras.callbacks.CSVLogger(keras_model_epoch_file, separator=',', append=False)
    history = model.fit(train_input, train_output, epochs=EPOCHS,validation_data=(val_input,val_output), verbose=2, callbacks=[NN_print_status(),csv_logger]);
    model.save(keras_model_file);

history=np.loadtxt(keras_model_epoch_file, delimiter=',',skiprows=1); plot_history(history);

[train_avg_percent_error,train_max_percent_error]=analyze_plot_error(train_output,train_input,model,'Training',True);
[test_avg_percent_error,test_max_percent_error]=analyze_plot_error(test_output,test_input,model,'Testing',True);
[val_avg_percent_error,val_max_percent_error]=analyze_plot_error(val_output,val_input,model,'Validation',True);

toc();