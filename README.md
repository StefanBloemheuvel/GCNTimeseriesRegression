## Code for the paper: Multivariate Time Series Regression with Graph Neural Networks

### Authors: Stefan Bloemheuvel, Jurgen van den Hoogen, Dario Jozinovic, Alberto Michelini and Martin Atzmueller

### This is the anonymized page that will be freely assessible in the future.
### This page is also under construction
--------------------------

#### Data
The data (too big to host on github itself) can be downloaded at: https://zenodo.org/record/5767221  <br /> 
In the data folder, the input_ci.npy file should be placed <br /> 
The input_cw.npy file should be placed in data/othernetwork <br /> 

#### Requirements
Tensorflow <br /> 
Spektral <br /> 
Networkx <br /> 

#### How to run

Run either main_cnn.py or main_gcn.py with the sys argument 'network1' or 'network2' in terminal, following the with 'nofeatures' or 'main' for the main version. Lastly, a number that serves as the random state for the split.<br /> 

Example: python main_gcn.py network1 main 1  <br /> 
If you want to also see how the graph is generated, graph_maker.py could be run as well. <br /> 


 <p align="center">
    <img src="./only_gnnblock.png", height="500">
 </p>
