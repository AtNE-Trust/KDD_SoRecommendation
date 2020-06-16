

from time import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
import networkx as nx
from general.graph import Graph


def read_file_all(config):
  """
  Read graph file and build graph object according to config argument

  :param config: a dictionary containing keys like
   'data_path', 'ftype', 'sep' ('\t'), 'comment' ('#'), 'directed' (False), 'weighted' (False), 'signed' (False)
    and optionally 'train_ratio', 'train_link_path', 'test_link_path', 'max_node'

  Returns
  -------
  graph object built from links
  """
  input_path = config['data_path']  # edge list to construct graph
  ftype = config['ftype']  # file type specifier
  sep = config['sep'] if 'sep' in config else '\t'  # separator for csv format
  comment = config['comment'] if 'comment' in config else '#'  # comment indicator for csv format
  directed = config['directed'] if 'directed' in config else True  # existence of direction on edges
  weighted = config['weighted'] if 'weighted' in config else False  # existence of weights on edges
  signed = config['signed'] if 'signed' in config else True  # existence of weight sign on edges

  print("Reading file")
  t0 = time()
  # Read file and build edge list
  if ftype == "mat":
    # read file with scipy.io.loadmat into coo matrix
    mat_var = loadmat(input_path)
    if "network" in mat_var:
      mat_matrix = mat_var["network"]
    else:
      mat_matrix = mat_var["Problem"]
      mat_matrix = mat_matrix[0][0][2]
    coo = mat_matrix.tocoo()

    links = np.vstack([coo.row, coo.col, coo.data]).T
  elif ftype == "gml":
    # read file with networkx.read_gml into networkx graph
    network = nx.read_gml(input_path)

    if weighted or signed:
      links = np.concatenate([[(s, t, network[s][t]['value']) for t in network[s].keys()] for s in network.adj.keys()])
    else:
      links = np.concatenate([[(s, t, 1.0) for t in network[s].keys()] for s in network.adj.keys()])
  else:
    # read file with pandas.read_csv
    if weighted or signed:
      usecols = [0, 1, 2]
      names = ['u1', 'u2', 'w']
    else:
      usecols = [0, 1]
      names = ['u1', 'u2']
    df = pd.read_csv(input_path, sep=sep, comment=comment, usecols=usecols, names=names)

    links = np.array(df)

  print("Preprocessing on link list")
  # According to configuration, filter edge list before build graph
  if 'cv_fold' in config:
    cv_fold = config['cv_fold']
    test_ratio = 1 / cv_fold
    print(test_ratio)
    np.random.shuffle(links)
    # if cv_fold is set, train-test set is built and return
    for i in range(cv_fold):
      mask = np.zeros(len(links), np.bool)
      mask[i * int(len(links) * test_ratio): (i + 1) * int(len(links) * test_ratio)] = 1
      test_links = links[mask, :]
      print(len(test_links))
      np.savetxt(config['link_path'] + '-test_links(t%.1f,cv%d).txt' % (1 - test_ratio, i), test_links, fmt='%d')
      mask = np.ones(len(links), np.bool)
      mask[i * int(len(links) * test_ratio): (i + 1) * int(len(links) * test_ratio)] = 0
      train_links = links[mask, :]
      print(len(train_links))
      np.savetxt(config['link_path'] + '-train_links(t%.1f,cv%d).txt' % (1 - test_ratio, i), train_links, fmt='%d')
    return
  elif 'train_ratio' in config:
    train_ratio = config['train_ratio']
    if train_ratio < 1:
      # if train ratio is set to be <1, separately save train links and test links as files and only keep train links
      np.random.shuffle(links)
      test_links = links[int(len(links) * train_ratio):, :]
      np.savetxt(config['test_link_path'] if 'test_link_path' in config else './test/test_links.txt', test_links,
                 fmt='%d')
      links = links[:int(len(links)*train_ratio), :]
      np.savetxt(config['train_link_path'] if 'train_link_path' in config else './test/train_links.txt', links,
                 fmt='%d')
  elif 'test_link_path' in config:
    # if test_link_path is set but train_ratio is not set, remove prespecified test links from edge list
    test_links = np.loadtxt(config['test_link_path'])
    test_links = test_links[:, :2]
    test_set = set((x, y) for x, y in test_links[:, :2])
    links = np.array([(x, y, w) for x, y, w in links if (x, y) not in test_set])
  elif 'max_node' in config:
    # if max_node is set, choose principal submatrix with max_node
    max_node = config['max_node']
    nodes = np.unique(links[:, :2])

    # union/find to get connected component in principal submatrix
    set_nodes = dict(zip(nodes, nodes))
    rank = dict(zip(nodes, np.zeros(len(nodes), dtype=np.int)))
    sel_edges = list()
    union_cnt = 0
    for i, (x, y) in enumerate(links[:, :2]):
      if x > max_node or y > max_node or x == y:
        continue
      if set_find(set_nodes, x) != set_find(set_nodes, y):
        set_merge(set_nodes, rank, x, y)
        union_cnt += 1
      sel_edges.append(i)
    links = links[np.array(sel_edges)]

    # only links within connected component
    set_idx = [set_find(set_nodes, node) for node in np.unique(links[:, :2])]
    counts = np.bincount(set_idx)
    keep_nodes = np.argwhere(np.array(set_idx) == np.argmax(counts))[:, 0]
    sel_edges = list()
    for i, (x, y) in enumerate(links[:, :2]):
      if x in keep_nodes and y in keep_nodes:
        sel_edges.append(i)
    links = links[np.array(sel_edges)]

  print("Building graph")
  # Build Graph object from list of links
  g = Graph(directed, weighted, signed)
  g.build_graph(links)
  g.make_consistent()
  g.preprocess()
  print("Loaded graph in {}s".format(time() - t0))

  return g


def set_find(set_nodes, i):
  """
  Set find implementation with path compression

  :param set_nodes: dictionary where key is node id and value is set containing node
  :param i: id of node

  Returns
  -------
  id of the set containing the node i
  """
  if i != set_nodes[i]:
    set_nodes[i] = set_find(set_nodes, set_nodes[i])
  return set_nodes[i]


def set_merge(set_nodes, rank, i, j):
  """
  Set merge implementation with union by rank

  :param set_nodes: dictionary where key is node id and value is set containing node
  :param rank: rank dictionary where key is set id and value is rank of the set
  :param i: id of the node contained in first set to merge
  :param j: id of the node contained in second set to merge
  """
  set_i = set_find(set_nodes, i)
  set_j = set_find(set_nodes, j)
  if set_i == set_j:
    return
  if rank[set_i] > rank[set_j]:
    set_nodes[set_j] = set_i
  else:
    set_nodes[set_i] = set_j
    if rank[set_i] == rank[set_j]:
      rank[set_j] += 1
