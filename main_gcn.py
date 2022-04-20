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
import sys
from helper_functions import *

network_choice = sys.argv[1]
model_chosen = sys.argv[2]
random_state_here = int(sys.argv[3])

def build_model(input_shape):

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    graph_input = layers.Input(shape=(39,39), name='graph_input')
    graph_features = layers.Input(shape=(39,2), name='graph_features')

    conv1 = layers.Conv1D(filters=32, kernel_size=125, strides=2,  activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='conv1')(wav_input)
    conv1 = layers.Conv1D(filters=64, kernel_size=125, strides=2,  activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='conv2')(conv1)
    conv1_new = tf.keras.layers.Reshape((39,conv1.shape[2] * conv1.shape[3]))(conv1)    

    if model_chosen == 'nofeatures':
        print('went for no features version')
    if model_chosen == 'main':
        print('went for features version')
        conv1_new = layers.concatenate(inputs=[conv1_new, graph_features], axis=2)

    conv1_new = GCNConv(64, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.0001))([conv1_new, graph_input])
    conv1_new = GCNConv(64, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(0.0001))([conv1_new, graph_input])

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
test_set_size = 0.2

if network_choice == 'network1':
    inputs = np.load('data/inputs_ci.npy', allow_pickle = True)
    targets = np.load('data/targets.npy', allow_pickle = True)
  
    graph_input = np.load('data/minmax_normalized_laplacian.npy', allow_pickle=True)
    graph_input = np.array([graph_input] * inputs.shape[0])

    graph_features = np.load('data/station_coords.npy', allow_pickle=True)
    graph_features = np.array([graph_features] * inputs.shape[0])

if network_choice == 'network2':
    inputs = np.load('data/othernetwork/inputs_cw.npy', allow_pickle = True)
    targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)
    
    graph_input = np.load('data/othernetwork/minmax_normalized_laplacian.npy', allow_pickle=True)
    graph_input = np.array([graph_input] * inputs.shape[0])

    graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
    graph_features = np.array([graph_features] * inputs.shape[0])

train_inputs, test_inputs, traingraphinput , testgraphinput, train_graphfeature, test_graphfeature, train_targets, testTargets = train_test_split(inputs,graph_input, graph_features, targets, test_size=test_set_size, random_state=random_state_here)
testInputs, testMaxes = normalize(test_inputs[:, :, :1000, :])        

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

    trainInputs, trainMaxes = normalize(trainInputsAll[:, :, :1000, :])
    valInputs, valMaxes = normalize(valInputsAll[:, :, :1000, :])

    model = build_model(valInputs[0].shape)

    iteration_checkpoint = keras.callbacks.ModelCheckpoint(
        f'models/graph_model_{network_choice}_iteration_{k}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )

    # print(model.summary())
    
    history = model.fit(x=[trainInputs,train_graphinput,train_graphfeatureinput], 
                        y=targets_to_list(trainTargets),#trainTargets[:,:,0],
            epochs=100, batch_size=20,
            validation_data=([valInputs,val_graphinput,val_graphfeatureinput], targets_to_list(valTargets)),verbose=0,callbacks=[es,iteration_checkpoint])#
    

    print()
    print('total number of epochs ran = ',len(history.history['loss']))
    print('Fold number:' + str(k))
    predictions = model.predict([testInputs,testgraphinput, test_graphfeature])

    new_predictions = np.array(predictions)
    new_predictions = np.swapaxes(new_predictions,0,2)
    new_predictions = np.swapaxes(new_predictions,0,1)
    
    np.save(f'saved_results/gcn_{network_choice}_iteration_{k}_targets.npy',testTargets)
    np.save(f'saved_results/gcn_{network_choice}_iteration_{k}_predictions.npy',new_predictions)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    def calculate_results(ground_truth,predictions):
        MSE = []
        for i in range(0,5):
            MSE.append(mean_squared_error(ground_truth[:,:,i], predictions[:,:,i]))
        MSE = np.array(MSE).mean()
        
        RMSE = []
        for i in range(0,5):
            RMSE.append(mean_squared_error(ground_truth[:,:,i], predictions[:,:,i], squared=False))
        RMSE = np.array(RMSE).mean()
        
        MAE = []
        for i in range(0,5):
            MAE.append(mean_absolute_error(ground_truth[:,:,i], predictions[:,:,i]))
        MAE = np.array(MAE).mean()
        
        mse_list.append(MSE)
        rmse_list.append(RMSE)
        mae_list.append(MAE)
        
    calculate_results(testTargets,new_predictions)

    # reset keras and tensorflow
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
        print(f'{print_time()},{sys.argv[0]},PGV,{network_choice},{mse_list[0]},{rmse_list[0]},{mae_list[0]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PGA,{network_choice},{mse_list[1]},{rmse_list[1]},{mae_list[1]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA03,{network_choice},{mse_list[2]},{rmse_list[2]},{mae_list[2]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA1,{network_choice},{mse_list[3]},{rmse_list[3]},{mae_list[3]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA3,{network_choice},{mse_list[4]},{rmse_list[4]},{mae_list[4]},{random_state_here}', file=text_file)




