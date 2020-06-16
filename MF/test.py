# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:56:25 2019

@author: 45016577
"""

import networkx as nx
import matplotlib.pyplot as plt
list2 = []
list1 = [[1,2,3],[2,3,4]]
for item in list1:
    tuple1 = tuple(item)
    list2.append(tuple1)
print(list2)
G = nx.DiGraph()
G.add_weighted_edges_from([(1,2,3),(2,3,0.2)])
print(G.get_edge_data(1,2))
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]
#节点位置
pos=nx.spring_layout(G) # positions for all nodes
#首先画出节点位置
# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)
#根据权重，实线为权值大的边，虚线为权值小的边
# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,width=6)
nx.draw_networkx_edges(G,pos,edgelist=esmall,alpha=0.5,width=3,edge_color="g",style="dashed")
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
nx.draw_networkx_edge_labels(G,pos,font_size=10,alpha=0.5,rotate=True);
#(G,pos = nx.random_layout(G)，node_color = 'b',edge_color = 'r',with_labels = True，font_size =18,node_size =20)
#plt.savefig("figure.png")
plt.axis("off")
plt.show()