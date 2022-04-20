from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime 
import numpy as np

seed = 1
import tensorflow
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
    return np.array(normalized), np.array(maxes)        


def targets_to_list(targets): # Houden
    targets = targets.transpose(2,0,1)

    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
        
    return targetList


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




