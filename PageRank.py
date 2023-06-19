import networkx as nx
import pandas as pd
import time
import numpy as np


start_time = time.time()

def Euclidean_Dist(df1, df2, cols):
    return np.linalg.norm(df1[cols].values - df2[cols].values,axis=1)

G= nx.fast_gnp_random_graph(5000,0.5,directed=True)
nodes_num = G.number_of_nodes()
#nx.draw(G,with_labels = True)
nodes = list(G.nodes())
alpha = 0.85
p = pd.DataFrame([[0.0]*nodes_num]*nodes_num)

for node in nodes:
    print (node)
    neighbors = list (G.predecessors(node))
    for n in neighbors:
        p[node][n] = 1/(G.out_degree(n))
    
p = p.T

for node in nodes:
    if p[node].sum() == 0:
        p[node] = [1/nodes_num]*nodes_num
        
r = pd.DataFrame([1/nodes_num]*nodes_num)
A = alpha*p + (1-alpha)*(p*0.0+1/nodes_num)
Ak = A
counter = 0
while sum(Euclidean_Dist(Ak,Ak.dot(A), Ak.columns))>0.000001:
    Ak = Ak.dot(A)
    counter+=1
    print('counter= ',counter)

rk = Ak.dot(r)
print("--- %s seconds ---" % (time.time() - start_time))
