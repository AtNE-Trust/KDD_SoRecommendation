
import tensorflow as tf
from multiprocessing import cpu_count
import threading
import time
import sys


class Side(object):
  """ SiDE model inference graph """

  def __init__(self, config, sess):
    """
    Initializer to set local variables, preload random walk data and build graph

    :param config: a dictionary containing keys like
     'num_walks', 'embed_dim', 'window_size', 'neg_sample_size', 'damping_factor', 'balance_factor',
      'regularization_param', 'batch_size', 'learning_rate', 'clip_norm',
       'epochs_to_train', 'summary_interval', 'save_interval', 'final_walk_path', 'embed_path'
    :param sess: tensorflow session to execute tensorflow operations
    """

    self.embed_dim = config['embed_dim']
    self.window_size = config['window_size']
    self.neg_sample_size = config['neg_sample_size']
    self.beta = config['damping_factor']
    self.gamma = config['balance_factor']
    self.regularization_param = config['regularization_param']
    self.batch_size = config['batch_size']
    self.learning_rate = config['learning_rate']
    self.clip_norm = config['clip_norm']
    self.epochs_to_train = config['epochs_to_train']
    self.summary_interval = config['summary_interval']
    self.save_interval = config['save_interval']
    self.concurrent_step = config['concurrent_step'] if 'concurrent_step' in config else cpu_count()
    self.final_walk_path = config['final_walk_path']
    self.embed_path = config['embed_path']

    self._sess = sess
    self._node2idx = dict()
    self._idx2node = list()
    self.build_graph()
    self.save_vocab()

  def build_graph(self):
    """
    Build computation graph using custom ops defined in side_kernels.cc
    """
    t0 = time.time()
    word2vec = tf.load_op_library("./embedding/side_ops.so")

    # preload random walk data and show statistics
    (words, counts, words_per_epoch, self._epoch, self._words_processed,
     self._examples, self._labels, self.num_pos, self.num_neg) = word2vec.skipgram_side(filename=self.final_walk_path,
                                                                                        batch_size=self.batch_size,
                                                                                        window_size=self.window_size)
    (self.vocab_words, self.vocab_counts, self.words_per_epoch) = self._sess.run([words, counts, words_per_epoch])
    self.vocab_size = len(self.vocab_words)
    print("read walk file pipeline done in %ds" % (time.time() - t0))

    self._idx2node = self.vocab_words
    for i, w in enumerate(self._idx2node):
      self._node2idx[w] = i

    self._W_target = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embed_dim],
                        - 0.5 / self.embed_dim, 0.5 / self.embed_dim),
      name="W_target")
    self._W_context = tf.Variable(
      tf.zeros([self.vocab_size, self.embed_dim]),
      name="W_context")
    tf.summary.histogram("target_weight", self._W_target)
    tf.summary.histogram("context_weight", self._W_context)

    self._b_out_pos = tf.Variable(tf.zeros([self.vocab_size]), name="b_out_pos")
    self._b_out_neg = tf.Variable(tf.zeros([self.vocab_size]), name="b_out_neg")
    self._b_in_pos = tf.Variable(tf.zeros([self.vocab_size]), name="b_in_pos")
    self._b_in_neg = tf.Variable(tf.zeros([self.vocab_size]), name="b_in_neg")
    tf.summary.histogram("positive_out_bias", self._b_out_pos)
    tf.summary.histogram("negative_out_bias", self._b_out_neg)
    tf.summary.histogram("positive_in_bias", self._b_in_pos)
    tf.summary.histogram("negative_in_bias", self._b_in_neg)

    self.multiplier = tf.multiply(
      tf.multiply(
        tf.pow(tf.constant([-1], dtype=tf.float32), tf.cast(self.num_neg, tf.float32)),
        tf.pow(tf.constant([self.beta], dtype=tf.float32), tf.cast(self.num_pos + self.num_neg - 1, tf.float32))),
      tf.pow(tf.constant([self.gamma], dtype=tf.float32), tf.cast(self.num_neg - 1, tf.float32))
      )
    self.global_step = tf.Variable(0, name="global_step")
    words_to_train = float(self.words_per_epoch * self.epochs_to_train)
    self._lr = self.learning_rate * tf.maximum(0.0001,
                                               1.0 - tf.cast(self._words_processed, tf.float32) / words_to_train)

    # define one step of training operation
    inc = self.global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      self._train = word2vec.neg_train_side(self._W_target,
                                            self._W_context,
                                            self._b_in_pos,
                                            self._b_in_neg,
                                            self._b_out_pos,
                                            self._b_out_neg,
                                            self._examples,
                                            self._labels,
                                            self._lr,
                                            self.multiplier,
                                            tf.constant(self.regularization_param),
                                            vocab_count=self.vocab_counts.tolist(),
                                            num_negative_samples=self.neg_sample_size)

    self._sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

  def save_vocab(self):
    """
    Save vocabulary file from statistics of random walk data
    """
    with open(self.embed_path + ".vocab", "w") as f:
      for i in range(self.vocab_size):
        vocab_word = tf.compat.as_text(self.vocab_words[i]).encode("utf-")
        f.write("%s %d \n" % (vocab_word, self.vocab_counts[i]))

  def _train_thread_body(self):
    """
    Function called by each threads
    Execute training operation as long as the current epoch is not changed
    """
    initial_epoch, prev_words = self._sess.run([self._epoch, self._words_processed])
    while True:
      _, epoch, words = self._sess.run([self._train, self._epoch, self._words_processed])
      if epoch != initial_epoch:
        break

  def train(self):
    """
    Train the side model using multi threads
    """
    initial_epoch, initial_words = self._sess.run([self._epoch, self._words_processed])
    self.summary = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter(self.embed_path, self._sess.graph)

    workers = []
    for _ in range(self.concurrent_step):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    # Print statistics while multi-threads execute the training updates
    last_words, last_time, last_summary_time, last_checkpoint_time = initial_words, time.time(), 0, 0
    while True:
      time.sleep(5)
      (epoch, step, words, lr) = self._sess.run(
        [self._epoch, self.global_step, self._words_processed, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
      print("Epoch %4d Step %8d: lr = %5.7f words/sec = %8.0f\r" % (epoch, step, lr, rate), end="")
      sys.stdout.flush()

      if now - last_summary_time > self.summary_interval:
        summary, global_step = self._sess.run([self.summary, self.global_step])
        self.writer.add_summary(summary, global_step)
        last_summary_time = now

      if epoch != initial_epoch:
        break
    for t in workers:
      t.join()
