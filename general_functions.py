#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:45:36 2020

@author: Dr. Shantanu Shahane
"""

import numpy as np
from matplotlib import pyplot as plt
import timeit
import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def get_time_column_header(surface_type,separation,surface_temperature):
    time_column_header=surface_type+'_'+str(separation)+'mm_'
    if surface_type=='SHP' :
        time_column_header=time_column_header + str(surface_temperature) + 'C Time[min]'
    elif surface_type=='R':
        time_column_header=time_column_header + str(surface_temperature) + 'C [min]'
    elif surface_type=='SHL':
        time_column_header=time_column_header + str(-surface_temperature) + 'C'
    else:
        string='Error: Unidentified surface type: ' + surface_type
        raise ValueError(string)
    if separation!=2 and separation!=4 and separation!=6 and separation!=8:
        string='Error: Unidentified surface type: ' + str(separation)
        raise ValueError(string)
    if surface_temperature!=-5 and surface_temperature!=-10 and surface_temperature!=-15:
        string='Error: Unidentified surface temperature: ' + str(surface_temperature)
        raise ValueError(string)
    return time_column_header

def get_data(raw_data,time_column_header, train_fraction,test_fraction,plot_flag):
#    time_minutes=raw_data.iloc[:,data_column_index];
    if not time_column_header in raw_data.columns:
        string='Error: Unidentified column header: ' + str(time_column_header)
        raise ValueError(string)
    time_minutes=raw_data[time_column_header]; time_minutes=pd.to_numeric(time_minutes, errors='coerce')
    time_minutes=time_minutes.to_numpy(); #time_minutes = time_minutes[~np.isnan(time_minutes)]
    frost_mm=raw_data.iloc[:,raw_data.columns.get_loc(time_column_header)+1];
    frost_mm=pd.to_numeric(frost_mm, errors='coerce')
    frost_mm=frost_mm.to_numpy(); #frost_mm = frost_mm[~np.isnan(frost_mm)]
    index=(~np.isnan(time_minutes) * ~np.isnan(frost_mm))
    time_minutes=time_minutes[index]; frost_mm=frost_mm[index];

    index=np.arange(len(time_minutes))
    np.random.shuffle(index)
    train_fraction=(int)(train_fraction*len(time_minutes)); test_fraction=(int)(test_fraction*len(time_minutes))
    val_fraction=len(time_minutes)-train_fraction-test_fraction
#    print(train_fraction,test_fraction,val_fraction)

    train_input=time_minutes[index[0:train_fraction]];
    train_output=frost_mm[index[0:train_fraction]];
    test_input=time_minutes[index[train_fraction:train_fraction+test_fraction]];
    test_output=frost_mm[index[train_fraction:train_fraction+test_fraction]];
    val_input=time_minutes[index[train_fraction+test_fraction:len(time_minutes)]];
    val_output=frost_mm[index[train_fraction+test_fraction:len(time_minutes)]];

    if plot_flag:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(time_minutes,frost_mm); axs[0, 0].set_title('Complete Data')
        axs[0, 1].scatter(train_input,train_output); axs[0, 1].set_title('Training Data')
        axs[1, 0].scatter(test_input,test_output); axs[1, 0].set_title('Testing Data')
        axs[1, 1].scatter(val_input,val_output); axs[1, 1].set_title('Validation Data')

    return time_minutes,frost_mm,train_input,train_output,test_input,test_output,val_input,val_output

def write_NN_hyperparameters(folder,time_column_header, l2_reg_lambda, learning_rate, EPOCHS, dropout_factor, n_hidden_layers, n_hidden_units):
    fname=folder+time_column_header+'/hyperparameters.csv'
    f = open(fname, 'w')
    f.write("l2_reg_lambda,%g\n" % (l2_reg_lambda))
    f.write("learning_rate,%g\n" % (learning_rate))
    f.write("dropout_factor,%g\n" % (dropout_factor))

    f.write("EPOCHS,%i\n" % (EPOCHS))
    f.write("n_hidden_layers,%i\n" % (n_hidden_layers))
    f.write("n_hidden_units,%i\n" % (n_hidden_units))
    f.close()

def build_NN_model(n_features,n_outputs,n_hidden_layers,n_hidden_units,l2_reg_lambda,learning_rate,dropout_factor):
    model = keras.Sequential();
    for i in range(n_hidden_layers):
        if i==0:
            model.add(keras.layers.Dense(n_hidden_units[i], activation='relu', input_shape=(n_features,), kernel_regularizer=keras.regularizers.l2(l2_reg_lambda))); #first hidden layer
            model.add(keras.layers.Dropout(dropout_factor))
        else:
            model.add(keras.layers.Dense(n_hidden_units[i], activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg_lambda))); #other hidden layers
            model.add(keras.layers.Dropout(dropout_factor));
    model.add(keras.layers.Dense(n_outputs, activation='linear')); #output layer
#    optimizer = tf.train.RMSPropOptimizer(learning_rate);
#    optimizer = keras.optimizers.SGD(lr=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)
#    optimizer = keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
#    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#    optimizer = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#    optimizer = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
    return model

class NN_print_status(keras.callbacks.Callback): # Display training progress by printing a single dot for each completed epoch
    def on_train_begin(self,logs):
        print('\n');
#        plt.figure();
    def on_epoch_end(self, epoch, logs):
#        plt.plot(epoch,(logs.get('loss')).reshape(-1,),'r--', label='Train Loss');
#        plt.legend();
#        if epoch % 100 == 0: print('')
        print(str(epoch+1))

def plot_history(history):
    plt.figure(); plt.xlabel('Epoch'); plt.ylabel('MAE or Loss');
    if isinstance(history,np.ndarray):
        plt.semilogy(history[:,0],history[:,1],'g--', label='Train Loss');
        plt.semilogy(history[:,0],history[:,2], 'g', label='Train mae');
        plt.semilogy(history[:,0],history[:,3],'r--', label='Val Loss');
        plt.semilogy(history[:,0],history[:,4], 'r',label='Val mae');
        plt.legend();
    else:
        plt.semilogy(history.epoch,np.array(history.history['loss']),'g--', label='Train Loss');
        plt.semilogy(history.epoch,np.array(history.history['mean_absolute_error']), 'g', label = 'Train mae');
        plt.semilogy(history.epoch,np.array(history.history['val_loss']),'r--', label='Val Loss');
        plt.semilogy(history.epoch,np.array(history.history['val_mean_absolute_error']),'r', label = 'Val mae');
        plt.legend(); #plt.ylim([0, 5])

def analyze_plot_error(output_exact,input1,model_NN,error_name,plot_flag,factor=None):
    output_NN = model_NN.predict(input1); output_NN=output_NN.reshape([-1,])
    error=abs(output_NN-output_exact)
    if factor is None:
        error=error/np.max(abs(output_exact));
    else:
        error=error/factor
    avg_percent_error=100.0*np.sum(error)/error.size; max_percent_error=100.0*np.max(error);
    print('\nOverall Relative %s Error: Average = %.2f Percent, Max = %.2f Percent' %(error_name,avg_percent_error,max_percent_error ));
    if plot_flag==True:
        plt.figure(); plt.plot(100.0*np.reshape(error,[-1,])); plt.xlabel('Sample Number'); plt.ylabel('Max Percent Relative Error'); plt.title(error_name+ ' Error');
    return avg_percent_error,max_percent_error

def superpose_plot(output_exact,input1,model_NN,folder,time_column_header):
    output_NN = model_NN.predict(input1); output_NN=output_NN.reshape([-1,])
    plt.figure(); plt.plot(input1,output_exact,'o-y', label="Experimental Data")
#    plt.scatter(input1, output_exact, s=80, facecolors='none', edgecolors='b', label="Experimental Data")
    plt.plot(input1,output_NN,'--k', linewidth=6, label="Neural Network Estimate")
    plt.xlabel('Time (min)'); plt.ylabel('Frost Thickness (mm)')
    plt.legend(loc="lower right"); plt.savefig(folder+time_column_header+'/superposed.png')