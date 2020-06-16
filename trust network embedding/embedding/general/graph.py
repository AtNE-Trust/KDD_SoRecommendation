

import numpy as np
from time import time


class Graph(object):
  """
  Graph class using nested dictionary
  Format: {src: {dst: weight}}
  If directed, also save in-links as {dst: {src: weight}}
  """

  def __init__(self, directed=False, weighted=False, signed=False):
    """
    Initializer to set graph properties and initialize the graph

    :param directed: existence of direction on edges
    :param weighted: existence of weights on edges
    :param signed: existence of weight sign on edges
    """
    self.directed = directed
    self.weighted = weighted
    self.signed = signed
    self.out_links = dict()
    self.in_links = dict()
    self.trans_prob = dict()
    self.freq = dict()

  def build_graph(self, lists_of_links):
    """
    Add links into the graph from lists of links

    :param lists_of_links: list or array of links with format (from_node_id, to_node_id, (optional) weight)
    """
    for link in lists_of_links:
      if self.weighted or self.signed:
        self.add_edge(int(link[0]), int(link[1]), float(link[2]))
      else:
        self.add_edge(int(link[0]), int(link[1]))

  def add_edge(self, fr, to, weight=1.0):
    """
    Add single link into appropriate dictionary

    :param fr: id of from node
    :param to: id of to node
    :param weight: weight on the edge (fr, to)
    """
    assert isinstance(fr, np.int), "invalid source node"
    assert isinstance(to, np.int), "invalid destination node"
    assert isinstance(weight, np.float), "invalid edge weight"
    if not self.weighted and not self.signed:
      weight = 1.0

    add_to_dbl_dict(self.out_links, fr, to, weight)
    if self.directed:
      # if directed, save in links
      add_to_dbl_dict(self.in_links, to, fr, weight)
    else:
      # if undirected, preserve symmetry
      add_to_dbl_dict(self.out_links, to, fr, weight)

  def make_consistent(self):
    """
    Make the graph consistent by removing self-loops and nodes with no links
    """
    loops_removed = 0
    remove_list_out = list()
    remove_list_in = list()

    # Remove self loop and save nodes with no out links
    for src in self.out_links:
      if src in self.out_links[src]:
        self.out_links[src].pop(src)
        loops_removed += 1
        if self.directed:
          self.in_links[src].pop(src)
      if len(self.out_links[src]) == 0:
        remove_list_out.append(src)

    # Remove node with no out link
    for node in remove_list_out:
      self.out_links.pop(node)

    # Save nodes with no in links
    if self.directed:
      for dst in self.in_links:
        if len(self.in_links[dst]) == 0:
          remove_list_in.append(dst)

    # Remove node with no in link
    for node in remove_list_in:
      self.in_links.pop(node)

    if loops_removed > 0:
      print("Removed {} self loops".format(loops_removed))
    if len(remove_list_out) > 0:
      print("Removed {} nodes with no out-links".format(len(remove_list_out)))
    if len(remove_list_in) > 0:
      print("Removed {} nodes with no in-links".format(len(remove_list_in)))

  def preprocess(self):
    """
    Build transition probability and node frequency tables
    Transition probability is used in random walk, thereby determined by out-links
    Node frequency is used in subsampling, thereby determined by in-links
    """
    # Build transition probability only when weighted
    # Unweighted probability from count for efficiency
    if self.weighted:
      for src, links in self.out_links.items():
        wgts = np.fromiter(links.values(), dtype=float)
        if self.signed:
          # Transition probability determined by absolute weight
          wgts = abs(wgts)
        norm = np.sum(wgts)
        self.trans_prob[src] = wgts / norm

    # Build in-degree frequency to calculate subsample rate
    if self.directed:
      # Directed graph freq estimated from in links
      freq_dict = self.in_links
    else:
      # Undirected graph freq estimated from total links
      freq_dict = self.out_links

    if self.weighted:
      if self.signed:
        # Frequency determined by absolute weight
        num_edges = np.sum([np.sum(np.abs(np.fromiter(links.values(), dtype=float))) for links in freq_dict.values()])
        self.freq = {node: np.sum(np.abs(np.fromiter(links.values(), dtype=float))) / num_edges
                     for node, links in freq_dict.items()}
      else:
        # Not applying absolute for unsigned
        num_edges = np.sum([np.sum(np.fromiter(links.values(), dtype=float)) for links in freq_dict.values()])
        self.freq = {node: np.sum(np.fromiter(links.values(), dtype=float)) / num_edges
                     for node, links in freq_dict.items()}
    else:
      # Unweighted probability from count for efficiency
      num_edges = np.sum([len(links.values()) for links in freq_dict.values()])
      self.freq = {node: len(links.values()) / num_edges for node, links in freq_dict.items()}

  def delete_deg1(self):
    """
    Delete nodes with degree 1 for optimization in embedding process and keep the parent information separately
    """
    t0 = time()
    if self.directed:
      # delete nodes with only 1 out-link and no in-link
      self.linked = dict()
      for node in self.out_links.keys():
        if len(self.out_links[node]) == 1:
          if node not in self.in_links:
            parent, = self.out_links[node].keys()
            if self.out_links[node][parent] > 0:
              self.linked[node] = parent
              self.in_links[parent].pop(node)
      for node, _ in self.linked.items():
        self.out_links.pop(node)

      # delete nodes with only 1 in-link and no out-link
      self.in_linked = dict()
      for node in self.in_links.keys():
        if len(self.in_links[node]) == 1:
          if node not in self.out_links:
            parent,  = self.in_links[node].keys()
            if self.in_links[node][parent] > 0:
              self.in_linked[node] = parent
              self.out_links[parent].pop(node)
      for node, _ in self.in_linked.items():
        self.in_links.pop(node)
    else:
      # delete nodes with only 1 link
      self.linked = dict()
      for node in self.out_links.keys():
        if len(self.out_links[node]) == 1:
          parent, = self.out_links[node].keys()
          if self.out_links[node][parent] > 0:
            self.linked[node] = parent
            self.out_links[parent].pop(node)
      for node, _ in self.linked.items():
        self.out_links.pop(node)

    # Redo make_consistent and preprocess
    self.make_consistent()
    self.trans_prob = dict()
    self.freq = dict()
    self.preprocess()
    print("Delete deg-1 nodes in {}s".format(time() - t0))

  @property
  def nodes(self):
    """ List of all nodes """
    return list(set(self.in_links.keys()).union(set(self.out_links.keys())))

  @property
  def edges(self):
    """ List of all edges """
    return [(fr, to, self.out_links[fr][to]) for fr in self.out_links.keys() for to in self.out_links[fr].keys()]

  @property
  def number_of_nodes(self):
    """ Number of all nodes """
    return len(self.nodes)

  @property
  def number_of_edges(self):
    """ Number of all edges """
    return len(self.edges)

  @property
  def has_edge(self, fr, to):
    """ Return whether an edge (fr, to) exists """
    return to in self.out_links[fr]


def add_to_dbl_dict(dbl_dict, key1, key2, value):
  """
  Helper function to add value in the nested dictionary with key1 and key2

  :param dbl_dict: nested dictionary
  :param key1: first key
  :param key2: second key
  :param value: value to insert
  """
  if key1 not in dbl_dict:
    dbl_dict[key1] = dict()
  elif not isinstance(dbl_dict[key1], dict):
    dbl_dict[key1] = dict()

  dbl_dict[key1][key2] = value
