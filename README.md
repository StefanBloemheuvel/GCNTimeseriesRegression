## Code for the paper: Multivariate Time Series Regression with Graph Neural Networks

### Authors: Stefan Bloemheuvel, Jurgen van den Hoogen, Dario Jozinovi, Alberto Michelini and Martin Atzmueller

### This is the anonymized page that will be freely assessible in the future.

--------------------------

#### Data
The data (too big to host on github itself) can be downloaden at: https://zenodo.org/record/5767221  <br /> 
In the data folder, the input_ci.npy file should be placed and renamed to input.npy <br /> 
The input_cw.npy file should be place in data/othernetwork  and also renamed to input.npy <br /> 

#### Requirements
Tensorflow <br /> 
Spektral <br /> 
Networkx <br /> 

#### How to run

Run either main_cnn.py or main_gcn.py with the sys argument 'network_1' or 'network_2' in terminal.<br /> 

If you want to also see how the graph is generated, graph_maker.py could be run as well. <br /> 
