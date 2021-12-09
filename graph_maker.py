#%%
import inspect
import networkx as nx
from networkx.classes.function import degree
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix
from networkx.readwrite import edgelist 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from spektral.transforms import normalize_adj
import os 

print(os.getcwd())


def graph_creator(station_choice, cutoff):
    print(station_choice, cutoff)
    if station_choice == 'network_1':
        stations = pd.read_pickle('data/chosenStations.pkl')
        stations = stations[['Station','Latitude','Longitude']]
        print(stations.shape)
        print(stations.head(2))
        
    else:
        stations = pd.read_csv('data/othernetwork/stationDists.csv')
        stations = stations[['sta','lat', 'lon']]
        stations.columns = ['Station','Latitude','Longitude']
        print(stations.shape)
        print(stations.head(2))
    
    station_coords = stations[['Latitude','Longitude']].values
    # station_coords = np.tile(station_coords, (50,1,1))
    if station_choice == 'network_1':
        np.save('data/station_coords.npy', station_coords)
    else:
        np.save('data/othernetwork/station_coords.npy', station_coords)

    graph = nx.Graph()

    for k in stations[['Station','Longitude','Latitude']].iterrows():
        graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))

    print(nx.info(graph))

    distances = []
    from geopy.distance import geodesic

    for idx1, itm1 in stations[['Station','Longitude','Latitude']].iterrows():
            for idx2, itm2 in stations[['Station','Longitude','Latitude']].iterrows():
                    pos1 = (itm1[1],itm1[2])
                    pos2 = (itm2[1],itm2[2])
                    distance = geodesic(pos1, pos2,).km #geopy distance
                    if distance != 0: # this filters out self-loops and also the edges between the artificial nodes
                        graph.add_edge(itm1[0], itm2[0], weight=distance, added_info=distance)
    print(nx.info(graph))

    names = []
    for i in graph.nodes():
        names.append(i)
    indexes = [i for i in range(0,39)]
    zip_iterator = zip(indexes, names)
    a_dictionary = dict(zip_iterator)

    edge_list = nx.to_pandas_edgelist(graph)
    edge_list['weight'] = (edge_list['weight'] - min(edge_list['weight'])) / (max(edge_list['weight']) - min(edge_list['weight']))
    edge_list['weight'] = 0.98 - edge_list['weight']
    adj = nx.from_pandas_edgelist(edge_list, edge_attr=['weight'])#,source=['source'], target=['target'])
    adj = pd.DataFrame(nx.adjacency_matrix(adj, weight='weight').todense())
    adj[adj < cutoff] = 0
    newgraph = nx.from_pandas_adjacency(adj)
    print()
    print(nx.info(newgraph))
    newgraph = nx.relabel_nodes(newgraph, a_dictionary)
    pos = nx.get_node_attributes(graph,'pos')
    nx.set_node_attributes(newgraph, nx.get_node_attributes(graph,'pos'),'pos')
    nx.set_edge_attributes(newgraph, nx.get_edge_attributes(graph,'added_info'),'added_info')
  
    edges1 = sorted(newgraph.edges(data=True),key= lambda x: x[2]['weight'],reverse=False)[0][0]
    edges2 = sorted(newgraph.edges(data=True),key= lambda x: x[2]['weight'],reverse=False)[0][1]
    edges3 = sorted(newgraph.edges(data=True),key= lambda x: x[2]['weight'],reverse=False)[0][2]
    
    print(f'edge with shortest weight = {edges1, edges2, edges3}, OG km = {graph[edges1][edges2]}')
    print(f'Average degree of the graph =  {np.mean([val for (node, val) in sorted(newgraph.degree(), key=lambda pair: pair[0])])}')

    degree_centralities = []
    for i in nx.degree_centrality(newgraph).values():
        degree_centralities.append(i)
    print('avg degree centrality = ',np.mean(degree_centralities))
    
    distances_og = []
    for i in newgraph.edges(data=True):
        distances_og.append(i[2]['added_info'])
    print('average distance og = ',np.array(distances_og).mean())

    adjacency = adj.copy()

    D = np.diag(np.sum(adjacency,axis=1))
    I = np.identity(adjacency.shape[0])
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = I - np.dot(D_inv_sqrt, adjacency).dot(D_inv_sqrt)

    normalized_laplacian_version = L.copy()

    if station_choice == 'network_1':
        np.save('data/minmax_normalized_laplacian.npy', normalized_laplacian_version)
    else:
        np.save('data/othernetwork/minmax_normalized_laplacian.npy', normalized_laplacian_version)

    return newgraph, pos
    
newgraph1, pos1 = graph_creator('network_1', 0.3) # optimal is 0.3
newgraph2, pos2 = graph_creator('network_2', 0.6) # optimal is 0.6
