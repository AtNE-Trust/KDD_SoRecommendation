

from time import time
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


def walk_on_memory(g, config):
  """
  Perform random walk simulation and hold in memory

  :param g: graph object where random walk occurs
  :param config: a dictionary containing keys like
   'num_walks', 'walk_length', 'window_size', 'subsample'

  Returns
  -------
  list of walks
  """
  num_walks = config['num_walks']  # the number of walk per each node
  walk_length = config['walk_length']  # the length of each walk
  window_size = config['window_size']  # the window size for neighborhood
  # subsample rate to delete frequent nodes in the random walk
  subsample = config['subsample'] if 'subsample' in config else 1e-3

  t0 = time()
  walks = []
  nodes = np.fromiter(g.out_links.keys(), dtype=np.int)
  global __vertex2str
  __vertex2str = {v: str(v) for v in g.nodes}
  __vertex2str["+"] = "+"
  __vertex2str["-"] = "-"

  # precalculate the number of pairs required for the walk starting from each node
  if g.directed:
    num_pairs_required = window_size * (walk_length - (window_size + 1) / 2)
    print("pairs required: %d" % num_pairs_required)
  else:
    num_pairs_required = window_size * (walk_length - (window_size + 1) / 2)

  if g.weighted:
    # For weighted graph, alias table preprocess for each node
    print("preprocess start")
    alias_nodes = preprocess_node_transition_probs(g)
    print("preprocess end in %ds" % (time() - t0))

    # Processing of random walk simulation
    for cnt in range(num_walks):
      if (cnt + 1) % 10 == 0:
        print(str(cnt + 1) + '/' + str(num_walks))
      np.random.shuffle(nodes)
      for node in nodes:
        num_pairs = 0
        while num_pairs < num_pairs_required:
          walks.append(random_walk_weighted(g, walk_length, node, np.random.RandomState(), subsample,
                                            alias_nodes=alias_nodes))
          num_pairs += ((len(walks[-1]) + 1) * (len(walks[-1]) - 1) / 8
                        if len(walks[-1]) < 2 * window_size - 1
                        else window_size * (len(walks[-1]) - window_size) / 2)
  else:
    # Processing of random walk simulation
    for cnt in range(num_walks):
      if (cnt + 1) % 10 == 0:
        print(str(cnt + 1) + '/' + str(num_walks))
      np.random.shuffle(nodes)
      for node in nodes:
        num_pairs = 0
        while num_pairs < num_pairs_required:
          walks.append(random_walk(g, walk_length, node, np.random.RandomState(), subsample))
          num_pairs += ((len(walks[-1]) + 1) * (len(walks[-1]) - 1) / 8
                        if len(walks[-1]) < 2 * window_size - 1
                        else window_size * (len(walks[-1]) - window_size) / 2)

  print("Walk on memory in {}s".format(time() - t0))
  return walks


def walk_to_disk(g, config):
  """
  Perform random walk simulation and save into files

  :param g: graph object where random walk occurs
  :param config: a dictionary containing keys like
   'num_walks', 'walk_length', 'window_size', 'walk_path', 'num_workers', 'subsample'

  Returns
  -------
  list of files containing random walks
  """
  num_walks = config['num_walks']  # the number of walk per each node
  walk_length = config['walk_length']  # the length of each walk
  window_size = config['window_size']  # the window size for neighborhood
  walk_path = config['walk_path']  # directory to save walk file
  # the number of processes for parallel processing
  num_workers = config['num_workers'] if 'num_workers' in config else cpu_count()
  # subsample rate to delete frequent nodes in the random walk
  subsample = config['subsample'] if 'subsample' in config else 1e-3

  t0 = time()
  global __current_graph
  global __alias_nodes
  global __alias_edges
  global __vertex2str
  __current_graph = g

  # For weighted graph, alias table preprocess for each node
  if g.weighted:
    print("preprocess start")
    __alias_nodes = preprocess_node_transition_probs(g)
    print("preprocess end in %ds" % (time() - t0))
  __vertex2str = {v: str(v) for v in g.nodes}
  __vertex2str["+"] = "+"
  __vertex2str["-"] = "-"
  files_list = ["{}_{}.walk".format(walk_path, str(x)) for x in range(num_workers)]

  # allocate random walks to different workers
  if num_walks <= num_workers:
    paths_per_worker = [1 for _ in range(num_walks)]
  else:
    paths_per_worker = [int(num_walks / num_workers) for _ in range(num_workers)]
    if num_walks - sum(paths_per_worker) > 0:
      for i in range(num_walks - sum(paths_per_worker)):
        paths_per_worker[i] += 1

  # different iteration function for different types of graph
  if g.weighted:
    iter_function = walk_on_memory_weighted_iter
  else:
    iter_function = walk_on_memory_default_iter
  if g.directed:
    num_pairs_required = window_size * (walk_length - (window_size + 1) / 2)
  else:
    num_pairs_required = window_size * (walk_length - (window_size + 1) / 2)
  args_list = [(ppw, walk_length, window_size, num_pairs_required, subsample, file, iter_function, seed)
               for seed, (file, ppw) in enumerate(zip(files_list, paths_per_worker))]

  # Parallel processing of random walk simulation
  t0 = time()
  files = list()
  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for file in executor.map(write_walks_to_disk, args_list):
      files.append(file)

  print("Walk on disk in {}s".format(time() - t0))
  return files


def write_walks_to_disk(args):
  """
  Write random walk into file

  :param args: arguments for random walk write

  Returns
  -------
  the name of file containing random walks
  """
  num_walks, walk_length, window_size, num_pairs_required, subsample, file_name, iter_function, seed = args
  rand = np.random.RandomState(seed)
  g = __current_graph
  with open(file_name, 'w') as fout:
    for walk in iter_function(g, num_walks, walk_length, window_size, num_pairs_required, subsample, rand):
      fout.write(u"{} &\n".format(u" ".join(walk)))
  return file_name


def walk_on_memory_default_iter(g, num_walks, walk_length, window_size, num_pairs_required, subsample, rand):
  """
  Iterator for random walk on unweighted network

  :param g: graph object where random walk occurs
  :param num_walks: the number of walk per each node
  :param walk_length: the length of each walk
  :param window_size: the window size for neighborhood
  :param num_pairs_required: the number of pairs required for random walks starting from each node
  :param subsample: subsample rate to delete frequent nodes in the random walk
  :param rand: the numpy random object

  Returns
  -------
  iterate over all walk created for assigned arguments
  """
  nodes = np.fromiter(g.out_links.keys(), dtype=np.int)
  for cnt in range(num_walks):
    rand.shuffle(nodes)
    for node in nodes:
      num_pairs = 0
      while num_pairs < num_pairs_required:
        walk = random_walk(g, walk_length, node, rand, subsample)
        yield walk
        num_pairs += ((len(walk) + 1) * (len(walk) - 1) / 8
                      if len(walk) < 2 * window_size - 1
                      else window_size * (len(walk) - window_size) / 2)


def walk_on_memory_weighted_iter(g, num_walks, walk_length, window_size, num_pairs_required, subsample, rand):
  """
  Iterator for random walk on weighted network

  :param g: graph object where random walk occurs
  :param num_walks: the number of walk per each node
  :param walk_length: the length of each walk
  :param window_size: the window size for neighborhood
  :param num_pairs_required: the number of pairs required for random walks starting from each node
  :param subsample: subsample rate to delete frequent nodes in the random walk
  :param rand: the numpy random object

  Returns
  -------
  iterate over all walk created for assigned arguments
  """
  nodes = np.fromiter(g.out_links.keys(), dtype=np.int)
  alias_nodes = __alias_nodes
  for cnt in range(num_walks):
    rand.shuffle(nodes)
    for node in nodes:
      num_pairs = 0
      while num_pairs < num_pairs_required:
        walk = random_walk_weighted(g, walk_length, node, rand, subsample, alias_nodes)
        yield walk
        num_pairs += ((len(walk) + 1) * (len(walk) - 1) / 8
                      if len(walk) < 2 * window_size - 1
                      else window_size * (len(walk) - window_size) / 2)


def random_walk(g, walk_length, start, rand, subsample):
  """
  Generate single random walk for unweighted network

  :param g: graph object where random walk occurs
  :param walk_length: the length of each walk
  :param start: the starting node for the random walk
  :param rand: the numpy random object
  :param subsample: subsample rate to delete frequent nodes in the random walk

  Returns
  -------
  list of nodes and signs defining random walk
  """
  walk = [start]

  cur = start
  sign = 1
  while len(walk) < 2 * walk_length - 1:
    if cur not in g.out_links:
      break
    nxt = rand.choice(list(g.out_links[cur].keys()))
    if rand.rand() < np.sqrt(subsample / g.freq[nxt]) or nxt not in g.out_links:
      walk.append("+" if sign * g.out_links[cur][nxt] > 0 else "-")
      walk.append(nxt)
      sign = 1
    elif g.out_links[cur][nxt] < 0:
      sign *= -1
    cur = nxt

  return [__vertex2str[node] for node in walk]


def random_walk_weighted(g, walk_length, start, rand, subsample, alias_nodes):
  """
  Generate single random walk for weighted network

  :param g: graph object where random walk occurs
  :param walk_length: the length of each walk
  :param start: the starting node for the random walk
  :param rand: the numpy random object
  :param subsample: subsample rate to delete frequent nodes in the random walk
  :param alias_nodes: preprocessed alias sampling table for each node

  Returns
  -------
  list of nodes and signs defining random walk
  """
  walk = [start]

  cur = start
  sign = 1
  while len(walk) < 2 * walk_length - 1:
    if cur not in g.out_links:
      break
    nxt = list(g.out_links[cur].keys())[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1], rand)]
    if rand.rand() < np.sqrt(subsample / g.freq[nxt]) or nxt not in g.out_links:
      walk.append("+" if sign * g.out_links[cur][nxt] > 0 else "-")
      walk.append(nxt)
      sign = 1
    elif g.out_links[cur][nxt] < 0:
      sign *= -1
    cur = nxt

  return [__vertex2str[node] for node in walk]


def preprocess_node_transition_probs(g):
  """
  Preprocess alias table for the deepwalk

  :param g: graph object to preprocess alias table

  Returns
  -------
  dictionary of alias tables where key is node and value is alias table of the node
  """
  # build alias table for each node
  t0 = time()
  alias_nodes = dict()
  for node in g.out_links.keys():
    alias_nodes[node] = get_alias_node(g, node)
  print(time() - t0)
  return alias_nodes


def get_alias_node(g, node):
  """
  Setup alias table for node

  :param g: graph object to preprocess alias table
  :param node: node to build alias table

  Returns
  -------
  alias table and probability table for node
  """
  return alias_setup(g.trans_prob[node])


def alias_setup(probs):
  """
  Build tables for the alias sampling

  :param probs: the probability distribution to build alias table

  Returns
  -------
  alias table and probability table for probs
  """
  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int)

  smaller = list()
  larger = list()
  for k, prob in enumerate(probs):
    q[k] = K*prob
    if q[k] < 1.0:
      smaller.append(k)
    else:
      larger.append(k)

  while len(smaller) > 0 and len(larger) > 0:
    small = smaller.pop()
    large = larger.pop()

    J[small] = large
    q[large] = q[large] + q[small] - 1.0
    if q[large] < 1.0:
      smaller.append(large)
    else:
      larger.append(large)

  return J, q


def alias_draw(J, q, rand):
  """
  Alias sampling from the table J and q

  :param J: alias table
  :param q: probability table
  :param rand: random object to sample random number

  Returns
  -------
  alias sampled node
  """
  K = len(J)

  k = int(np.floor(rand.rand() * K))
  if rand.rand() < q[k]:
    return k
  else:
    return J[k]
