#!/usr/bin/python
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf
import os

import reader
import model
import utils
from collections import namedtuple
import json
import glob


# TODO: see this https://github.com/kentonl/ran/blob/bcee2c80633168f65b69187486ece32ba19ec9db/text8/char_train.py
# TODO: use GPU!
# TODO: execute test verify to track improvements in the epochs

'''
    Small config:
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

    Medium config:
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

    Large config:
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
'''

WORK_DIR = '/data/www/asimov/data/poet/'

nn_config = {
    'init_scale': 0.04,
    'max_grad_norm': 10,
    'num_layers': 2,
    'num_steps': 35,
    'hidden_size': 720,
    'keep_prob': .35,
    'batch_size': 32,
    'vocab_size': 10000

}

# With AdamOptimizer i need to use a way smaller learning rate such as 0.003
# https://github.com/OpenNMT/OpenNMT-py/issues/676
train_config = {
    'max_max_epoch': 140,
    'max_epoch': 90,
    # 'learning_rate': 0.003,
    'learning_rate': 0.003,
    'lr_decay': .95
}

summary_writer = tf.summary.FileWriter(WORK_DIR)


def get_data(data_path, dataset):
  raw_data = reader.text8_raw_data(data_path)
  return reader, raw_data


def run_epoch(session, m, data, eval_op, num_layers, is_training=False):

    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    train_perplexity = 0

    # state = tf.get_default_session.run(m.initial_state)
    state = [(x[0].eval(), x[1].eval()) for x in m.initial_state]

    for step, (x_data, y_data) in enumerate(reader.train_iterator(data, m.batch_size, m.num_steps)):

        feed_dict = {m.input_data: x_data, m.targets: y_data, m.is_training: is_training}
        feed_dict.update({m.initial_state[i]: state[i] for i in range(num_layers)})

        cost, state, _ = session.run([m.cost, m.final_state, eval_op], feed_dict)

        costs += cost
        iters += m.num_steps

        print("%.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / epoch_size, np.exp(costs / iters),
               iters * m.batch_size / (time.time() - start_time)))

        train_perplexity = np.exp(costs / iters)

    return train_perplexity


def main():

    # cleanup input dir
    ret = input('Are you sure you want to clean %s [yes|no] ' % (WORK_DIR,))
    if ret == 'yes':
        for f in glob.glob(os.path.join(WORK_DIR, '*')):
            if not f.endswith('.txt'):
                os.remove(f)
                print(f + ' deleted')

    config = namedtuple('TrainConfig', train_config.keys())(*train_config.values())


    model_config = namedtuple('ModelConfig', nn_config.keys())(*nn_config.values())
    # model_val_config = namedtuple('ModelConfig', nn_config.keys())(*nn_config.values())
    # model_test_config = namedtuple('ModelConfig', test_config.keys())(*test_config.values())

    with open(os.path.join(WORK_DIR, 'config.json'), 'w', encoding="utf8") as fh:
        json.dump(nn_config, fh)

    proc = reader.TextProcessor.from_file(os.path.join(WORK_DIR, 'input.txt'))

    proc.create_vocab(model_config.vocab_size)
    # proc.create_vocab_test(model_config.vocab_size)

    train_data = proc.get_vector()
    # test_data = proc.get_vector_test()

    np.save(os.path.join(WORK_DIR, 'vocab.npy'), np.array(list(proc.id2word)))
    proc.save_converted(os.path.join(WORK_DIR, 'input.conv.txt'))

    # gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    tfSession = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with tf.Graph().as_default(), tfSession as session:
        initializer = tf.random_uniform_initializer(-model_config.init_scale,
                                                    model_config.init_scale)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = model.Model(is_training=True, config=model_config)

        # with tf.variable_scope("model", reuse=True, initializer=initializer):
        #     mvalid = model.Model(is_training=False, config=model_val_config)
        #     mtest = model.Model(is_training=False, config=model_test_config)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        for i in range(config.max_max_epoch):

            # TODO: i need a better algorithm to auto-decay learning rate value
            # -> https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10

            """
            Learning rate is a hyper-parameter that controls how much 
            we are adjusting the weights of our network with respect the loss gradient.
            """
            # lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            # m.assign_lr(session, config.learning_rate * lr_decay)

            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            lr = config.learning_rate * lr_decay
            m.assign_lr(session, lr)

            print("\r\nEpoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op, model_config.num_layers, is_training=True)

            # print("Testing on non-batched Valid ...")
            # valid_perplexity = run_epoch(session, mvalid, train_data, tf.no_op(),
            # model_config.num_layers, is_training=False)
            # print("Full Valid Perplexity: %.3f, Bits: %.3f" % (valid_perplexity, np.log2(valid_perplexity)))
            #
            # print("Testing on non-batched Test ...")value.tag
            # test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(),
            # model_config.num_layers, is_training=False)
            # print("Full Test Perplexity: %.3f, Bits: %.3f" % (test_perplexity, np.log2(test_perplexity)))

            """
            [ Perplexity ]
            --------------
            Perplexity is a measurement of how well a probability distribution 
            or probability model predicts a sample. It may be used to compare probability models. 
            A low perplexity indicates the probability distribution is good at predicting the sample.
            """
            print("=> [%d] Train Perplexity: %.3f" % (i + 1, train_perplexity))

            summary = tf.Summary()

            value_perplexity = summary.value.add()
            value_perplexity.simple_value = train_perplexity
            value_perplexity.tag = "NLP Perplexity"
            value_lr = summary.value.add()
            value_lr.simple_value = lr
            value_lr.tag = "Learning Rate"
            value_lr_opt = summary.value.add()
            value_lr_opt.simple_value = m.get_lr_optimized(session)
            value_lr_opt.tag = "Learning Rate Optimized"
            summary_writer.add_summary(summary, i + 1)
            summary_writer.flush()

            ckp_path = os.path.join(WORK_DIR, 'model.ckpt')
            saver.save(session, ckp_path, global_step=i)

            summary_writer.add_graph(session.graph, global_step=1)
            summary_writer.flush()


if __name__ == "__main__":
    main()