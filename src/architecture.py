import tensorflow as tf
from src.tf_frankenstein.normalization import BatchNorm


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Architecture:
    def __init__(self, class_cardinality, vocab_size, embedding_size, n_recurrent_units, learning_rate,
                 name="architecture"):
        self.embedding_size = embedding_size
        self.n_recurrent_units = n_recurrent_units
        self.class_cardinality = class_cardinality
        self.vocab_size = vocab_size
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.name = name
        self.define_computation_graph()

        # Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        tf.reset_default_graph()
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        with tf.variable_scope("Placeholders"):
            comment_in = tf.placeholder(dtype=tf.int32, shape=(None, None), name="comment_in")
            is_train = tf.placeholder(dtype=tf.bool, shape=None, name="is_train")
            target = tf.placeholder(dtype=tf.int32, shape=(None, self.class_cardinality), name="target")
            acc_dev = tf.placeholder(dtype=tf.float32, shape=None, name="acc_dev")
            loss_dev = tf.placeholder(dtype=tf.float32, shape=None, name="loss_dev")
            keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="dropout_keep_prob")
            return ({"comment_in": comment_in, "target": target, "keep_prob": keep_prob,
                     "is_train": is_train, "acc_dev": acc_dev, "loss_dev": loss_dev})

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            embedding = tf.get_variable("word_embeddings", [self.vocab_size, self.embedding_size])
            x = tf.nn.embedding_lookup(embedding, self.placeholders.comment_in)
            x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.dropout(x=x, keep_prob=self.placeholders.keep_prob, name="dropout_embedding")
            recurrent_cell = tf.nn.rnn_cell.GRUCell(self.n_recurrent_units, activation=tf.nn.relu)
            outputs, states = tf.nn.dynamic_rnn(recurrent_cell, x, dtype=tf.float32, scope="recurrency")
            x = outputs[:, -1, :]
            x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.dropout(x=x, keep_prob=self.placeholders.keep_prob, name="dropout_rnn")
            output = tf.layers.dense(x, self.class_cardinality, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name="dense_1")
            return ({"output": output})

    def define_losses(self):
        with tf.variable_scope("Losses"):
            sigmoid_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.placeholders.target, tf.float32),
                                                                 logits=self.core_model.output,
                                                                 name="softmax")
            return ({"sigmoid_ce": sigmoid_ce})

    def define_optimizers(self):
        with tf.variable_scope("Optimization"):
            op = self.optimizer.minimize(self.losses.sigmoid_ce)
            return ({"op": op})

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            loss = tf.reduce_mean(self.losses.sigmoid_ce)
            train_scalar_probes = {"loss": loss}
            train_performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in train_scalar_probes.items()]
            train_performance_scalar = tf.summary.merge(train_performance_scalar)

            dev_scalar_probes = {"loss_dev": self.placeholders.loss_dev}
            dev_performance_scalar = [tf.summary.scalar(k, v, family=self.name) for k, v in dev_scalar_probes.items()]
            dev_performance_scalar = tf.summary.merge(dev_performance_scalar)
            return ({"loss": loss, "s_tr": train_performance_scalar, "s_de": dev_performance_scalar})
