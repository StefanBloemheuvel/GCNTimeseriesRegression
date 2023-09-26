# Graph Neural Networks for Multivariate Time Series Regression with Application to Seismic Data

### Authors: Stefan Bloemheuvel, Jurgen van den Hoogen, Dario Jozinovic, Alberto Michelini and Martin Atzmueller

--------------------------
 <p align="center">
    <img src="./only_gnnblock.png", height="500">
 </p>

## Data
The data (too big to host on github itself) can be downloaded at: https://zenodo.org/record/5767221  <br /> 
In the data folder, the input_ci.npy file should be placed <br /> 
The input_cw.npy file should be placed in data/othernetwork <br /> 

## Requirements
geopy==2.2.0 <br /> 
keras==2.8.0 <br /> 
networkx==2.7.1 <br /> 
numba==0.56.2 <br /> 
numpy==1.22.3 <br /> 
scikit-learn==1.0.2 <br /> 
scipy==1.8.0 <br /> 
sklearn==0.0 <br /> 
spektral==1.1.0 <br /> 
tensorflow==2.8.0 <br /> 

## How to run
Run either main_cnn.py or main_gcn.py with the sys argument 'network1' or 'network2' in terminal, following the with 'nofeatures' or 'main' for the main version. Lastly, a number that serves as the random state for the split.<br /> 


Example:
```
$ python main_gcn.py network1 main 1
```

Here, 'network1' refers to the CI network, 'main' refers to running the main experiment and '1' refers to a seed which can be used.

  <p align="center">
    <img src="./procedure.png" width="100%" height="100%"
 </p>

## Cite

If you compare with, build on, or use aspects of this work, please cite the following:
```
@article{bloemheuvel2022graph,
  title={Graph neural networks for multivariate time series regression with application to seismic data},
  author={Bloemheuvel, Stefan and van den Hoogen, Jurgen and Jozinovic, Dario and Michelini, Alberto and Atzmueller, Martin},
  journal={International Journal of Data Science and Analytics},
  pages={1--16},
  year={2022},
  publisher={Springer}
}
```
