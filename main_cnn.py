

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
    
    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)
    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)
    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5),  activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)
    
    conv1 = layers.Flatten()(conv1)
    conv1 = layers.Dropout(0.4, seed=seed)(conv1)
    # conv1 = 
    meta_input = layers.Input(shape=(1,), name='meta_input')
    meta = layers.Dense(1)(meta_input)

    graph_features = layers.Input(shape=(39,2), name='graph_features')
    graph_features_flattened = layers.Flatten()(graph_features)

    merged = layers.concatenate(inputs=[conv1, meta, graph_features_flattened])
    merged = layers.Dense(128)(merged)

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, meta_input,graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    
    rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.)

    # final_model.compile(optimizer=rmsprop, loss='mse', metrics=['mse'])
    final_model.compile(optimizer=rmsprop, loss='mse')#, metrics=['mse'])
    
    return final_model

from tensorflow import keras

es = keras.callbacks.EarlyStopping(patience=10, verbose=1, min_delta=0.001, monitor='val_loss', mode='min',baseline=None, restore_best_weights=True)

import tensorflow as tf


import sys
network_choice = str(sys.argv[1])
#%%
def main():

    if network_choice == 'network_1':
        inputs = np.load('data/inputs_ci.npy', allow_pickle = True)
        targets = np.load('data/targets.npy', allow_pickle = True)
        meta = np.load('data/meta.npy', allow_pickle = True)

        graph_features = np.load('data/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
        print(graph_features.shape)
        print(graph_features)
    if network_choice == 'network_2':
        
        inputs = np.load('data/othernetwork/inputs_cw.npy', allow_pickle = True)[0:265]
        targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)[0:265]
        meta = np.load('data/othernetwork/meta.npy', allow_pickle = True)[0:265]

        graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
        print(graph_features.shape)
        print(graph_features)



    import math
    length_size_min = inputs.shape[0] / 5
    print(f"size of length_size_min = {length_size_min}")
    length_size_min = math.floor(length_size_min)
    print(f"size of length_size_min = {length_size_min}")

    length_size_max = inputs.shape[0] -(inputs.shape[0] / 5)
    print(f"size of length_size_max = {length_size_max}")
    length_size_max = math.floor(length_size_max)
    print(f"size of length_size_max = {length_size_max}")
    inputsK, targetsK, metaK = k_fold_split(inputs, targets, meta)
    
    mse_scores_pgv = []
    mse_scores_pga = []
    mse_scores_psa_03s = []
    mse_scores_psa_1s = []
    mse_scores_psa_3s = []


    for k in range(0,5):

        keras.backend.clear_session()
        tf.keras.backend.clear_session()

        trainInputsAll, trainTargets, trainMeta, testInputsAll, testTargets, testMeta = merge_splits(inputsK, targetsK, metaK, k)
    
        trainInputs, trainMaxes = normalize(trainInputsAll[:, :, :1000, :]) # 100 samples per second
        testInputs, testMaxes = normalize(testInputsAll[:, :, :1000, :])
    
        train_graphfeatureinput = graph_features[0:length_size_max,:,:]
        test_graphfeatureinput = graph_features[0:length_size_min,:,:]


        model = build_model(testInputs[0].shape)
    
        iteration_checkpoint = keras.callbacks.ModelCheckpoint(
            f'models/cnn_model_{network_choice}_iteration_{k}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )

        print(model.summary())
        history = model.fit(x=[trainInputs, trainMaxes, train_graphfeatureinput], 
                            y=targets_to_list(trainTargets),#trainTargets[:,:,0],
              epochs=100, batch_size=30,
              validation_data=([testInputs, testMaxes,test_graphfeatureinput], targets_to_list(testTargets)),verbose=0, callbacks=[es,iteration_checkpoint])#
       

        print(f"total parameters in this model: {model.count_params():,.2f}")
        print('total number of epochs ran = ',len(history.history['loss']))
        print('Fold number:' + str(k))
        print('Loss: ',history.history['loss'][-1])
        print('val_loss: ',history.history['val_loss'][-1])


        predictions = model.predict([testInputs, testMaxes,test_graphfeatureinput])
#         np.save(f'data/cnn_testtargets_{network_choice}.npy', testTargets)
#         np.save(f'data/cnn_predictions_{network_choice}.npy', predictions)

#         np.save(f'data/boxplots/cnn_predictions_{network_choice}_{k}.npy', predictions)
#         np.save(f'data/boxplots/cnn_testtargets_{network_choice}_{k}.npy', testTargets)


#         np.save(f'data/cnn_testmeta_{network_choice}.npy', testMeta)

        fileName = 'data/res/SManalysis/kFoldML' + str(k) + '.csv'
        fileNameGMPE = 'data/res/SManalysis/kFoldGMPE' + str(k) + '.csv'

        print('MSE = ',np.square(np.subtract(predictions[1], testTargets[:,:,1])).mean())
    

        mse_scores_pga.append(np.round(np.square(np.subtract(np.array(predictions)[0,:,:], testTargets[:,:,0])).mean(), 4))
        mse_scores_pgv.append(np.round(np.square(np.subtract(predictions[1], testTargets[:,:,1])).mean(), 4))

        mse_scores_psa_03s.append(np.round(np.square(np.subtract(predictions[2], testTargets[:,:,2])).mean(), 4))
        mse_scores_psa_1s.append(np.round(np.square(np.subtract(predictions[3], testTargets[:,:,3])).mean(), 4))
        mse_scores_psa_3s.append(np.round(np.square(np.subtract(predictions[4], testTargets[:,:,4])).mean(), 4))
        

        keras.backend.clear_session()
        tf.keras.backend.clear_session()

    mean_pgv = np.array(mse_scores_pgv).mean()
    mean_pga = np.array(mse_scores_pga).mean()
    mean_psa_03s = np.array(mse_scores_psa_03s).mean()
    mean_psa_1s = np.array(mse_scores_psa_1s).mean()
    mean_psa_3s = np.array(mse_scores_psa_3s).mean()
    # mean_5 = str(np.array(mse_scores_5).mean())

 
    all_scores = str(np.array([mean_pgv,mean_pga,mean_psa_03s,mean_psa_1s,mean_psa_3s]).mean())

    with open("mse_results.csv", "a") as text_file:
        print(print_time() + ',' +'CNN'+ ',' + network_choice + ',' + str(mean_pgv) + ',' + str(mean_pga) + ',' + str(mean_psa_03s) + ',' + str(mean_psa_1s) + ',' + str(mean_psa_3s)  + ',' + str(all_scores) , file=text_file)


if __name__== "__main__" :
    main()

# %%
