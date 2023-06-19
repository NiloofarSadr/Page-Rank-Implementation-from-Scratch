import networkx as nx
import pandas as pd
import time
import numpy as np
import scipy.sparse as sp
import sklearn.metrics.pairwise
start_time = time.time()



#path = 'D:\\نظریه ریاضی سیستم ها\\MTS_HW3_Sadrolhefazi_830400072\\'
data = pd.read_csv('web-Google.txt',delimiter="\t",header = None)
print('data ro khooond!')

G= nx.DiGraph()
G.add_edges_from(list(data.itertuples(index=False, name=None)))
print('G dorost shod!!!!')
nodes_num = G.number_of_nodes()
#nx.draw(G,with_labels = True)
nodes = list(G.nodes())

#alpha = 0.8
#alpha = 0.85
alpha = 0.9

mat = sp.dok_matrix((916428,916428), dtype=np.float16)

counter = 0
for node in nodes:
    print (counter)
    counter+=1
    neighbors = list (G.predecessors(node))
    for n in neighbors:
        mat[node,n] = 1/(G.out_degree(n))
    if G.out_degree(node) ==0:
        for n in nodes:
            mat[n,node] = 1/(nodes_num)

mat = mat.tocsr()
print('csr')
r = np.array([1/nodes_num]*nodes_num)
print('r dorost shod')
A = alpha*mat 
B = A
b = B.nonzero()
for i in range(len(b[0])):
    B[b[0],b[1]] = 1/nodes_num
print('A*alpha ham shod')
A += (1-alpha)*B

Ak = A
counter = 0
while sum(sum(sklearn.metrics.pairwise.pairwise_distances(Ak,Ak*A)))>0.000001:
    Ak = Ak*A
    counter+=1
    print('counter= ',counter)

rk = Ak*r
print("--- %s seconds ---" % (time.time() - start_time))
