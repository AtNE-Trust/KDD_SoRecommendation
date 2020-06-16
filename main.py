

import argparse
import sys, os
from general import read_graph
from embedding import walk, train
import fileinput
from time import time

cur_dir = os.path.dirname(os.path.realpath(__file__))


def parse_args():
  """
  Parse arguments and pass as arguments object

  Returns
  -------
  arguments object
  """
  parser = argparse.ArgumentParser('side')
  # read_config
  parser.add_argument('--dataset', nargs='?', default='gama', help='Dataset name (gama)')
  parser.add_argument('--network-file', nargs='?', default='./graph/out.ucidata-gama',
                      help='Input network file (./graph/out.ucidata-gama)')
  parser.add_argument('--ftype', nargs='?', default='csv', help='Input file format (csv)')
  parser.add_argument('--sep', nargs='?', default='\t', help='Input separator (\t)')
  parser.add_argument('--comment', nargs='?', default='%', help='Input file comment indicator (%)')
  parser.add_argument('--walk-path', nargs='?', default='./walk/', help='Output path for walk (./walk/)')
  parser.add_argument('--embed-path', nargs='?', default='./output/', help='Output path for embedding (./output/)')

  parser.add_argument('--directed', dest='directed', action='store_false', default=True, help='directed graph')
  parser.add_argument('--weighted', dest='weighted', action='store_true', default=False, help='weighted graph')
  parser.add_argument('--signed', dest='signed', action='store_false', default=True, help='signed graph')

  parser.add_argument('--deg1', dest='deg1', action='store_false', default=True, help='parametrized walk')
  parser.add_argument('--subsample', type=float, default=1e-3, help='Input parameter')

  # walk_config
  parser.add_argument('--mem-max-walks', type=int, default=10000,
                      help='Maximum number of walks to process in memory (100000000)')
  parser.add_argument('--num-walks', type=int, default=80, help='Number of walks per node (80)')
  parser.add_argument('--walk-length', type=int, default=40, help='Length of walks (40)')
  parser.add_argument('--window-size', type=int, default=5, help='Number of context to train (5)')

  # embed_config
  parser.add_argument('--embed-dim', type=int, default=128, help='Dimension of embedding (128)')
  parser.add_argument('--neg-sample-size', type=int, default=20, help='Number of negative samples to train (20)')
  parser.add_argument('--regularization-param', type=float, default=0.01, help='Regularization parameter (0.01)')
  parser.add_argument('--batch-size', type=int, default=16, help='Size of batch to train in 1 iter (16)')
  parser.add_argument('--learning-rate', type=float, default=0.025, help='Learning rate (0.025)')
  parser.add_argument('--clip-norm', type=float, default=5.0, help='Gradient norm clipping (5.0)')
  parser.add_argument('--epochs-to-train', type=int, default=1, help='Number of epochs to train (1)')
  parser.add_argument('--summary-interval', type=int, default=100, help='Number of iteration between summary (100)')
  parser.add_argument('--save-interval', type=int, default=1000, help='Number of iteration between save (1000)')

  return parser.parse_args()


def main(args):
  """
  main function implementing trust network embedding

  :param args: arguments object
  """
  t0 = time()
  # define configuration for read, walk and embed
  read_config = dict()
  walk_config = dict()
  embed_config = dict()
  
  read_config['data_path'] = args.network_file
  read_config['ftype'] = args.ftype
  read_config['sep'] = args.sep
  read_config['comment'] = args.comment
  walk_config['walk_path'] = args.walk_path + args.dataset
  embed_config['final_walk_path'] = args.walk_path + args.dataset + '.walk'
  embed_config['embed_path'] = args.embed_path + args.dataset

  read_config['directed'] = args.directed
  read_config['weighted'] = args.weighted
  read_config['signed'] = args.signed

  walk_config['num_walks'] = args.num_walks
  walk_config['walk_length'] = args.walk_length
  walk_config['window_size'] = args.window_size
  walk_config['subsample'] = args.subsample

  embed_config['embed_dim'] = args.embed_dim
  embed_config['window_size'] = args.window_size
  embed_config['neg_sample_size'] = args.neg_sample_size
  embed_config['damping_factor'] = 1
  embed_config['balance_factor'] = 1
  embed_config['regularization_param'] = args.regularization_param
  embed_config['batch_size'] = args.batch_size
  embed_config['learning_rate'] = args.learning_rate
  embed_config['clip_norm'] = args.clip_norm
  embed_config['epochs_to_train'] = args.epochs_to_train
  embed_config['summary_interval'] = args.summary_interval
  embed_config['save_interval'] = args.save_interval

  # read files and build graph object
  g = read_graph.read_file_all(read_config)

  if args.deg1:
    # delete deg1 nodes for optimization
    g.delete_deg1()
    with open(embed_config['embed_path'] + ".deg1", "w") as f:
      for deleted, parent in g.linked.items():
        f.write("%d %d\n" % (deleted, parent))
    if g.directed:
      with open(embed_config['embed_path'] + ".indeg1", "w") as f:
        for deleted, parent in g.in_linked.items():
          f.write("%d %d\n" % (deleted, parent))


  # Random walk sampling phase
  data_size = g.number_of_nodes * args.num_walks * args.walk_length
  # walk on memory or disk according to walk size
  if data_size < args.mem_max_walks:
    print("walk on memory")
    walks = walk.walk_on_memory(g, walk_config)
    with open(embed_config['final_walk_path'], 'w') as fout:
      for w in walks:
        fout.write(u"{} &\n".format(u" ".join(w)))
    del walks
  else:
    print("walk on disk")
    files = walk.walk_to_disk(g, walk_config)
    with open(embed_config['final_walk_path'], 'w') as fout, fileinput.input(files) as fin:
      for line in fin:
        fout.write(line)
    for fin in files:
      os.remove(fin)
  print("Finished walks in %ds" % (time() - t0))

  train.train(embed_config)
  print("Finished training in %ds" % (time() - t0))


if __name__ == '__main__':
  args_ = parse_args()
  sys.exit(main(args_))
