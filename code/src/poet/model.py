import tensorflow as tf
from tensorflow.python.ops import rnn as tfrnn


class Model(object):

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.optimizer = None

        self._input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='input_data')
        self._targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name="targets")

        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(axis=1, num_or_size_splits=num_steps, value=inputs)]
        outputs, state = tfrnn.static_rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable('softmax_w', [size, vocab_size])
        softmax_b = tf.get_variable('softmax_b', [vocab_size])
        self._logits = logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)

        # Gradient Descent Optimizer is not compatible with Google TPU (Tensor Processing Unit)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = self.optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def get_lr_optimized(self, session):
        return session.run(self.optimizer._lr)

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def logits(self):
        return self._logits