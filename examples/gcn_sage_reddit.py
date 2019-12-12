import tensorflow as tf
import tf_euler

class MeanAggregator(tf_euler.layers.Layer):
  def __init__(self, dim, activation=tf.nn.relu):
    super(MeanAggregator, self).__init__()
    self.neigh_layer = tf_euler.layers.Dense(
        dim, activation=activation, use_bias=False)

  def call(self, neigh_embedding):
    agg_embedding = tf.reduce_mean(neigh_embedding, axis=1)
    from_neighs = self.neigh_layer(agg_embedding)
    return from_neighs

class SageEncoder(tf_euler.layers.Layer):
  def __init__(self, metapath, fanouts, dim, feature_idx, feature_dim):
    super(SageEncoder, self).__init__()
    self.metapath = metapath
    self.fanouts = fanouts
    self.num_layers = len(metapath)

    self.feature_idx = feature_idx
    self.feature_dim = feature_dim

    self.aggregators = []
    for layer in range(self.num_layers):
      activation = tf.nn.relu if layer < self.num_layers - 1 else None
      self.aggregators.append(MeanAggregator(dim, activation=activation))
    self.dims = [feature_dim] + [dim] * self.num_layers

  def call(self, inputs):
    samples = tf_euler.sample_fanout(inputs, self.metapath, self.fanouts)[0]
    hidden = [
        tf_euler.get_dense_feature(sample,
                                   [self.feature_idx], [self.feature_dim])[0]
        for sample in samples]
    for layer in range(self.num_layers):
      aggregator = self.aggregators[layer]
      next_hidden = []
      for hop in range(self.num_layers - layer):
        neigh_shape = [-1, self.fanouts[hop], self.dims[layer]]
        h = aggregator(tf.reshape(hidden[hop + 1], neigh_shape))
        next_hidden.append(h)
      hidden = next_hidden
    return hidden[0]

class GraphSage(tf_euler.layers.Layer):
  def __init__(self, label_idx, label_dim,
               metapath, fanouts, dim, feature_idx, feature_dim):
    super(GraphSage, self).__init__()
    self.label_idx = label_idx
    self.label_dim = label_dim
    self.encoder = SageEncoder(metapath, fanouts, dim, feature_idx, feature_dim)
    self.predict_layer = tf_euler.layers.Dense(label_dim)

  def call(self, inputs):
    nodes, labels = self.sampler(inputs)
    embedding = self.encoder(nodes)
    loss = self.decoder(embedding, labels)
    return (embedding, loss)

  def sampler(self, inputs):
    labels = tf_euler.get_dense_feature(inputs, [self.label_idx],
                                                [self.label_dim])[0]
    return inputs, labels

  def decoder(self, embedding, labels):
    logits = self.predict_layer(embedding)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #predictions = tf.floor(tf.nn.sigmoid(logits) + 0.5)
    #f1 = tf_euler.metrics.f1_score(labels, predictions)
    return tf.reduce_mean(loss)#, f1

tf_euler.initialize_embedded_graph('reddit')

source = tf_euler.sample_node(1000, 0)
source.set_shape([1000])

model = GraphSage(0, 41, [[0], [0]], [4, 4], 64, 1, 602)
_, loss = model(source)

global_step = tf.train.get_or_create_global_step()
train_op = tf.train.AdamOptimizer(0.03).minimize(loss, global_step)

tf.logging.set_verbosity(tf.logging.INFO)
with tf.train.MonitoredTrainingSession(
  hooks=[
      tf.train.LoggingTensorHook({'step': global_step,
                                  'loss': loss}, 100),
      tf.train.StopAtStepHook(2000)
  ]) as sess:
  while not sess.should_stop():
    sess.run(train_op)
