
#%%
import os
import pandas as pd

import numpy as np
# import obspy
# import xml.etree.ElementTree
# import urllib.request
import matplotlib.pyplot as plt
# import scipy
import json
import h5py

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split

import datetime

def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S")


def normalize(inputs): # Houden
    maxes = []
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        maxes.append(maks)
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
#    maxes = np.reshape(np.array(maxes), (len(maxes), 1))
    return np.array(normalized), np.array(maxes)        

def targets_to_list(targets): # Houden
    targets = targets.transpose(2,0,1)

    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
        
    return targetList


seed = 1
import tensorflow
def k_fold_split(inputs, targets, meta): # houden
    meta = np.stack(meta)

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
    meta = meta[p]
    
    ind = int(len(inputs)/5)
    inputsK = []
    targetsK = []
    metaK = []
    for i in range(0,5-1):
        inputsK.append(inputs[i*ind:(i+1)*ind])
        targetsK.append(targets[i*ind:(i+1)*ind])
        metaK.append(meta[i*ind:(i+1)*ind])
    
    inputsK.append(inputs[(i+1)*ind:])
    targetsK.append(targets[(i+1)*ind:])
    metaK.append(meta[(i+1)*ind:])    
    
    return inputsK, targetsK, metaK
        
def merge_splits(inputs, targets, meta, k): # houden
    if k != 0:
        z=0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
        metaTrain = meta[z]
    else:
        z=1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
        metaTrain = meta[z]

    for i in range(z+1, 5):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))
            metaTrain = np.concatenate((metaTrain, meta[i]))
    
    return inputsTrain, targetsTrain, metaTrain, inputs[k], targets[k], meta[k]


def build_model(input_shape): # houden

    reg_const = 0.0001
    activation_func = 'relu'

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    meta_input = layers.Input(shape=(1,), name='meta_input')
    
    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)
    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)
    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5),  activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)
    
    conv1 = layers.Flatten()(conv1)
    conv1 = layers.Dropout(0.4, seed=seed)(conv1)
    # conv1 = 
    # meta = layers.Dense(1)(meta_input)

    graph_features = layers.Input(shape=(39,2), name='graph_features')
    graph_features_flattened = layers.Flatten()(graph_features)

    merged = layers.concatenate(inputs=[conv1, graph_features_flattened])
    merged = layers.Dense(128)(merged)

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input,meta_input, graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    # final_model = models.Model(inputs=[wav_input,graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=None, decay=0.)

    # final_model.compile(optimizer=rmsprop, loss='mse', metrics=['mse'])
    final_model.compile(optimizer=rmsprop, loss='mse')#, metrics=['mse'])
    
    return final_model

from tensorflow import keras

es = keras.callbacks.EarlyStopping(patience=10, verbose=0, min_delta=0.001, monitor='val_loss', mode='min',baseline=None, restore_best_weights=True)

import tensorflow as tf


import sys
# network_choice = 'network_1'
network_choice = sys.argv[1]


#%%
if network_choice == 'network_1':
    test_set_size = 0.2
    inputs = np.load('data/inputs_ci.npy', allow_pickle = True)
    targets = np.load('data/targets.npy', allow_pickle = True)
    meta = np.load('data/meta.npy', allow_pickle = True)

    graph_features = np.load('data/station_coords.npy', allow_pickle=True)
    graph_features = np.array([graph_features] * inputs.shape[0])

if network_choice == 'network_2':
    test_set_size = 0.2
    inputs = np.load('data/othernetwork/inputs_cw.npy', allow_pickle = True)
    targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)
    meta = np.load('data/othernetwork/meta.npy', allow_pickle = True)

    graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
    graph_features = np.array([graph_features] * inputs.shape[0])
    
import random
random_state_here = int(sys.argv[2])

train_inputs, test_inputs, train_graphfeature, test_graphfeature, train_targets, testTargets = train_test_split(inputs, graph_features, targets, test_size=test_set_size, random_state=random_state_here)
testInputs, testMaxes = normalize(test_inputs[:, :, :1000, :])        

print('train inputs = ',train_inputs.shape)
# print('traingraphinputs = ',traingraphinput.shape)
print('traingraphfeature = ',train_graphfeature.shape)
print('traintargets = ',train_targets.shape)

print('test_inputs = ',test_inputs.shape)
# print('testgraphinput = ',testgraphinput.shape)
print('testgraph feature = ',test_graphfeature.shape)
print('testtargets = ',testTargets.shape)

import math
# print(train_inputs.shape)
print()
length_size_min = int((train_inputs.shape[0] / 5))
print(f"size of length_size_min = {length_size_min}")
# length_size_min = math.floor(length_size_min)
print(f"size of length_size_min = {length_size_min}")

length_size_max = int((train_inputs.shape[0]) + -(train_inputs.shape[0] / 5))
print(f"size of length_size_max = {length_size_max}")
# length_size_max = math.ceil(length_size_max)
print(f"size of length_size_max = {length_size_max}")
inputsK, targetsK, metaK = k_fold_split(train_inputs, train_targets, meta)

mse_list = []
rmse_list = []
mae_list = []
rsquared_list = []
maxerror_list = []
euclideanerror_list = []
mape_list = []

for k in range(0,5):

    keras.backend.clear_session()
    tf.keras.backend.clear_session()

    trainInputsAll, trainTargets, trainMeta, valInputsAll, valTargets, testMeta = merge_splits(inputsK, targetsK, metaK, k)

    print('hier kijken')
    print('traininputsall = ',trainInputsAll.shape)
    print(trainTargets.shape)
    print(trainMeta.shape)
    print(valInputsAll.shape)
    print(valTargets.shape)
    print(testMeta.shape)

    trainInputs, trainMaxes = normalize(trainInputsAll[:, :, :1000, :]) # 100 samples per second
    valInputs, valMaxes = normalize(valInputsAll[:, :, :1000, :])
    print(trainInputs.shape)

    # train_graphfeatureinput = train_graphfeature[0:length_size_max,:,:]
    # val_graphfeatureinput = train_graphfeature[0:length_size_min,:,:]

    train_graphfeatureinput = train_graphfeature[0:trainInputsAll.shape[0],:,:]
    val_graphfeatureinput = train_graphfeature[0:valInputsAll.shape[0],:,:]


    model = build_model(valInputs[0].shape)

    iteration_checkpoint = keras.callbacks.ModelCheckpoint(
        f'models/cnn_model_{network_choice}_iteration_{k}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )

    # print(model.summary())
    history = model.fit(x=[trainInputs,trainMaxes, train_graphfeatureinput], 
                        y=targets_to_list(trainTargets),#trainTargets[:,:,0],
            epochs=100, batch_size=20,
            validation_data=([valInputs,valMaxes,val_graphfeatureinput], targets_to_list(valTargets)),verbose=0, callbacks=[es,iteration_checkpoint])#
    

    print(f"total parameters in this model: {model.count_params():,.2f}")
    print('total number of epochs ran = ',len(history.history['loss']))
    print('Fold number:' + str(k))
    print('Loss: ',history.history['loss'][-1])
    print('val_loss: ',history.history['val_loss'][-1])


    predictions = model.predict([testInputs,testMaxes,test_graphfeature])

    new_predictions = np.array(predictions)
    new_predictions = np.swapaxes(new_predictions,0,2)
    new_predictions = np.swapaxes(new_predictions,0,1)

    # np.save(f'saved_results/cnn_{network_choice}_iteration_{k}_targets.npy',testTargets)
    # np.save(f'saved_results/cnn_{network_choice}_iteration_{k}_predictions.npy',new_predictions)


    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,max_error,mean_absolute_percentage_error
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
    
    RSQUARED = []
    for i in range(0,5):
        RSQUARED.append(r2_score(testTargets[:,:,i], new_predictions[:,:,i]))
    print('RSQUARED = ',np.array(RSQUARED).mean())
    RSQUARED = np.array(RSQUARED).mean()

    MAPE = []
    for i in range(0,5):
        MAPE.append(mean_absolute_percentage_error(testTargets[:,:,i], new_predictions[:,:,i]))
    MAPE = np.array(MAPE).mean()


    MAX_ERROR = []
    for i in range(0,5):
        MAX_ERROR.append(max_error(testTargets[:,:,i].flatten(), new_predictions[:,:,i].flatten()))
    MAX_ERROR = np.array(MAX_ERROR).mean()


    def dist(x,y):   
        return np.sqrt(np.sum((x-y)**2))

    EUCLIDEAN = []
    for i in range(0,5):
        EUCLIDEAN.append(dist(testTargets[:,:,i], new_predictions[:,:,i]))
    print('euclidean = ',np.array(EUCLIDEAN).mean())
    EUCLIDEAN = np.array(EUCLIDEAN).mean()

    mse_list.append(MSE)
    rmse_list.append(RMSE)
    mae_list.append(MAE)
    rsquared_list.append(RSQUARED)

    maxerror_list.append(MAX_ERROR)
    euclideanerror_list.append(EUCLIDEAN)
    mape_list.append(MAPE)


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
print('rsquared score = ',np.array(rsquared_list).mean())
print('max error score = ',np.array(maxerror_list).mean())
print('mape error score = ',np.array(mape_list).mean())


print('euclidean score = ',np.array(euclideanerror_list).mean())

#%%
print(np.array(mse_list).shape)
print(np.array(rmse_list).shape)
print(np.array(mae_list).shape)
print(np.array(rsquared_list).shape)
print(np.array(maxerror_list).shape)
print(np.array(mape_list).shape)
print(np.array(euclideanerror_list).shape)

#%%

# with open("new_all_results.csv", "a") as text_file:
#         print(print_time() + ',' +'GCNnofeatures'+ ',' + network_choice + ',' + str(np.array(mse_list).mean()) + ',' + str(np.array(rmse_list).mean()) + ',' + str(np.array(mae_list).mean()) + ',' + str(np.array(rsquared_list).mean()) + ',' + str(np.array(maxerror_list).mean()) +',' + str(np.array(euclideanerror_list).mean()) + ',' +  str(random_state_here), file=text_file)

with open("new_all_results.csv", "a") as text_file:
        # print(print_time() + ',' +'GCNnofeatures'+ ',' + network_choice + ',' + str(np.array(mse_list)[0]) + ',' + str(np.array(rmse_list)[0]) + ',' + str(np.array(mae_list)[0]) + ',' + str(np.array(rsquared_list).mean()) + ',' + str(np.array(maxerror_list).mean()) +',' + str(np.array(euclideanerror_list).mean()) + ',' +  str(random_state_here), file=text_file)

        print(f'{print_time()},{sys.argv[0]},PGV,{network_choice},{mse_list[0]},{rmse_list[0]},{mae_list[0]},{rsquared_list[0]},{maxerror_list[0]},{euclideanerror_list[0]},{mape_list[0]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PGA,{network_choice},{mse_list[1]},{rmse_list[1]},{mae_list[1]},{rsquared_list[1]},{maxerror_list[1]},{euclideanerror_list[1]},{mape_list[1]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA03,{network_choice},{mse_list[2]},{rmse_list[2]},{mae_list[2]},{rsquared_list[2]},{maxerror_list[2]},{euclideanerror_list[2]},{mape_list[2]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA1,{network_choice},{mse_list[3]},{rmse_list[3]},{mae_list[3]},{rsquared_list[3]},{maxerror_list[3]},{euclideanerror_list[3]},{mape_list[3]},{random_state_here}', file=text_file)
        print(f'{print_time()},{sys.argv[0]},PSA3,{network_choice},{mse_list[4]},{rmse_list[4]},{mae_list[4]},{rsquared_list[4]},{maxerror_list[4]},{euclideanerror_list[4]},{mape_list[4]},{random_state_here}', file=text_file)

#%%
