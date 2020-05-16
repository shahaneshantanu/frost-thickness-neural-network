#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:25:34 2020

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
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large', 'figure.figsize': (20, 10), 'axes.labelsize': 'x-large',        'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
tic();

folder='Neural Networks/'
file='Raw Data.xlsx'
surface_type='SHP'; #options: 'SHL', 'SHP', 'R'
separation=8; #options: 2, 4, 6, 8
surface_temperature=-5; #options: -5, -10, -15

time_column_header=get_time_column_header(surface_type,separation,surface_temperature)

if not os.path.exists(folder+time_column_header):
    string='Unable to find folder: '+folder+time_column_header
    raise ValueError(string)

raw_data=pd.read_excel(folder+file,skiprows=1);
time_minutes,frost_mm,_,_,_,_,_,_=get_data(raw_data, time_column_header, 1, 0, False);
keras_model_file=folder+time_column_header+'/keras.h5'
if os.path.isfile(keras_model_file): #trained model exists
    model=keras.models.load_model(keras_model_file); #model.summary();
else: #model is not yet trained; throw error
    string='Unable to find the Keras neural network file: '+keras_model_file
    raise ValueError(string)

[_,_]=analyze_plot_error(frost_mm,time_minutes,model,'',False);
superpose_plot(frost_mm,time_minutes,model,folder,time_column_header)

toc();