#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

from spektral.layers import GCNConv
from tensorflow.keras.layers import *
from spektral.utils import gcn_filter
from sklearn.model_selection import train_test_split
import datetime
from spektral.layers import GlobalAvgPool, GlobalMaxPool

import sys 

network_choice = sys.argv[1]
model_chosen = sys.argv[2]
random_state_here = int(sys.argv[3])

print('model = ',model_chosen)
print('network = ',network_choice)
print('random state = ',random_state_here)

seed = 1
def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S")  

def normalize(inputs): # Houden
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized)      

def targets_to_list(targets): # Houden
    targets = targets.transpose(2,0,1)

    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
        
    return targetList


seed = 1
import tensorflow
def k_fold_split(inputs, targets): # houden

    # make sure everything is seeded
    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    np.random.permutation(seed)
    tensorflow.random.set_seed(seed)
    
    p = np.random.permutation(len(targets))
    
    print('min of p = ',np.array(p)[50:100].min())
    print('max of p = ',np.array(p)[50:100].max())
    print('mean of p = ',np.array(p)[50:100].mean())
    inputs = inputs[p]
    targets = targets[p]

    
    ind = int(len(inputs)/5)
    inputsK = []
    targetsK = []

    for i in range(0,5-1):
        inputsK.append(inputs[i*ind:(i+1)*ind])
        targetsK.append(targets[i*ind:(i+1)*ind])

    
    inputsK.append(inputs[(i+1)*ind:])
    targetsK.append(targets[(i+1)*ind:])
  
    
    return inputsK, targetsK
        
def merge_splits(inputs, targets, k): # houden
    if k != 0:
        z=0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
    else:
        z=1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]

    for i in range(z+1, 5):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))
    
    return inputsTrain, targetsTrain, inputs[k], targets[k]

def build_model(input_shape): # houden

    reg_const = 0.0001
    activation_func = 'relu'

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    graph_input = layers.Input(shape=(39,39), name='graph_input')
    graph_features = layers.Input(shape=(39,2), name='graph_features')

    conv1 = layers.Conv1D(filters=32, kernel_size=125, strides=2,  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const), name='conv1')(wav_input)
    conv1 = layers.Conv1D(filters=64, kernel_size=125, strides=2,  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const), name='conv2')(conv1)

    conv1_new = tf.keras.layers.Reshape((39,conv1.shape[2] * conv1.shape[3]))(conv1)    
    if model_chosen == 'nofeatures':
        print('went for no features version')
    if model_chosen == 'main':

        print('went for features version')
        conv1_new = layers.concatenate(inputs=[conv1_new, graph_features], axis=2)

    conv1_new = GCNConv(64, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])
    conv1_new = GCNConv(64, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])

    conv1_new = layers.Flatten()(conv1_new)
    conv1_new = layers.Dropout(0.4, seed=seed)(conv1_new)

    merged = layers.Dense(128)(conv1_new)

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, graph_input, graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=None, decay=0.)
    final_model.compile(optimizer=rmsprop, loss='mse')
    
    return final_model

from tensorflow import keras

es = keras.callbacks.EarlyStopping(patience=10, verbose=0, min_delta=0.001, monitor='val_loss', mode='min',baseline=None, restore_best_weights=True)

import sys

#%%

if network_choice == 'network1':
    test_set_size = 0.2
    inputs = np.load('data/inputs_ci.npy', allow_pickle = True)
    targets = np.load('data/targets.npy', allow_pickle = True)
    
    graph_input = np.load('data/minmax_normalized_laplacian.npy', allow_pickle=True)
    graph_input = np.array([graph_input] * inputs.shape[0])

    graph_features = np.load('data/station_coords.npy', allow_pickle=True)
    graph_features = np.array([graph_features] * inputs.shape[0])

if network_choice == 'network2':
    test_set_size = 0.2
    inputs = np.load('data/othernetwork/inputs_cw.npy', allow_pickle = True)
    targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)
    
    graph_input = np.load('data/othernetwork/minmax_normalized_laplacian.npy', allow_pickle=True)
    
    graph_input = np.array([graph_input] * inputs.shape[0])

    graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
    graph_features = np.array([graph_features] * inputs.shape[0])

train_inputs, test_inputs, traingraphinput , testgraphinput, train_graphfeature, test_graphfeature, train_targets, testTargets = train_test_split(inputs,graph_input, graph_features, targets, test_size=test_set_size, random_state=random_state_here)
testInputs = normalize(test_inputs[:, :, :1000, :])        

import math


inputsK, targetsK = k_fold_split(train_inputs, train_targets)




mse_list = []
rmse_list = []
mae_list = []


for k in range(0,5):
    keras.backend.clear_session()
    tf.keras.backend.clear_session()

    trainInputsAll, trainTargets, valInputsAll, valTargets = merge_splits(inputsK, targetsK, k)

    train_graphinput = traingraphinput[0:trainInputsAll.shape[0],:,:]
    train_graphfeatureinput = train_graphfeature[0:trainInputsAll.shape[0],:,:]

    val_graphinput = traingraphinput[0:valInputsAll.shape[0],:,:]
    val_graphfeatureinput = train_graphfeature[0:valInputsAll.shape[0],:,:]

    if network_choice == 'network1':
        trainInputs = normalize(trainInputsAll[:, :, :1000, :])
        valInputs = normalize(valInputsAll[:, :, :1000, :])

    if network_choice == 'network2':
        trainInputs = normalize(trainInputsAll[:, :, :, :])
        valInputs = normalize(valInputsAll[:, :, :, :])

    model = build_model(valInputs[0].shape)

    iteration_checkpoint = keras.callbacks.ModelCheckpoint(
        f'models/graph_model_{network_choice}_iteration_{k}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )

    print(model.summary())
    
    history = model.fit(x=[trainInputs,train_graphinput,train_graphfeatureinput], 
                        y=targets_to_list(trainTargets),
            epochs=100, batch_size=20,
            validation_data=([valInputs,val_graphinput,val_graphfeatureinput], targets_to_list(valTargets)),verbose=0,callbacks=[es,iteration_checkpoint])#
    
    print()
    print('total number of epochs ran = ',len(history.history['loss']))
    print('Fold number:' + str(k))
    predictions = model.predict([testInputs,testgraphinput, test_graphfeature])

    new_predictions = np.array(predictions)
    new_predictions = np.swapaxes(new_predictions,0,2)
    new_predictions = np.swapaxes(new_predictions,0,1)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    MSE = []
    for i in range(0,5):
        MSE.append(mean_squared_error(testTargets[:,:,i], new_predictions[:,:,i]))
    print('mse = ',np.array(MSE).mean())
    MSE = np.array(MSE).mean()
    
    RMSE = []
    for i in range(0,5):
        RMSE.append(mean_squared_error(testTargets[:,:,i], new_predictions[:,:,i], squared=False))
    print('rmse = ',np.array(RMSE).mean())
    RMSE = np.array(RMSE).mean()
    
    MAE = []
    for i in range(0,5):
        MAE.append(mean_absolute_error(testTargets[:,:,i], new_predictions[:,:,i]))
    print('MAE = ',np.array(MAE).mean())
    MAE = np.array(MAE).mean()
    

    mse_list.append(MSE)
    rmse_list.append(RMSE)
    mae_list.append(MAE)

    keras.backend.clear_session()
    tf.keras.backend.clear_session()

print('-')
print('-')
print('-')
print('-')
print('all averages = ')
print('mse score = ',np.array(mse_list).mean())
print('rmse score = ',np.array(rmse_list).mean())
print('mae score = ',np.array(mae_list).mean())

with open("githubresults.csv", "a") as text_file:
        print(f'{print_time()},{sys.argv[0]},PGV,{network_choice},{model_chosen},{mse_list[0]},{rmse_list[0]},{mae_list[0]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PGA,{network_choice},{model_chosen},{mse_list[1]},{rmse_list[1]},{mae_list[1]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA03,{network_choice},{model_chosen},{mse_list[2]},{rmse_list[2]},{mae_list[2]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA1,{network_choice},{model_chosen},{mse_list[3]},{rmse_list[3]},{mae_list[3]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA3,{network_choice},{model_chosen},{mse_list[4]},{rmse_list[4]},{mae_list[4]},{random_state_here}', file=text_file)

