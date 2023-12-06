#%%
import numpy as np
import torch as pt
import time
# from hgraph import Tree
from hgraph_jit import hierarchical_graph as hgraph
# from hgraph_np import hierarchical_graph as ohgraph
from math import log2

#%%
n_ppe = 10
dtype = np.float32
ax = np.linspace(-4,4,n_ppe, dtype=dtype)
n_particles = n_ppe**3
v = 2 * np.random.random((n_particles,3), dtype=dtype) - 1
f = 2 * np.random.random((n_particles,3), dtype=dtype) - 1
m = np.ones((n_particles,1), dtype=dtype)
x = np.stack(np.meshgrid(ax,ax,ax), 3).reshape(-1,3)
dev = 'cpu'
nodes = np.concatenate((m,x,v,f), axis=1)
print(nodes.shape)

# %% compare vectorized class with old hierarchical_graph
levels = 5

start = time.time_ns()
jgraph = hgraph(nodes, levels=levels)
print("Time to build jit graph (incl compile):", (time.time_ns()-start)/1000000)
print("\n\n")
start = time.time_ns()
jgraph = hgraph(nodes, levels=levels)
print("Time to build jit graph:", (time.time_ns()-start)/1000000)
# start_o = time.time_ns()
# ograph = ohgraph(pt.tensor(nodes), levels=levels)
# print("Time to build old graph:", (time.time_ns()-start_o)/1000000)

# start_n = time.time_ns()
# tree = Tree(int(max(2, log2(n_ppe))), device=dev)
# tinit = time.time_ns()-start_n
# ngraph = tree.hierarchical_graph(nodes)
# tn = time.time_ns()-start_n
# print("Time to build new graph new:", tn/1000000)
# print("Time to build new graph without init:", (tn-tinit)/1000000)


# %%
