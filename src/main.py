import json

import tensorflow as tf
import numpy as np
import pandas as pd
from src.data_tools import load_train_data, get_batcher
from src.tensorflow_utilities import start_tensorflow_session, get_summary_writer, TensorFlowSaver
from src.architecture import Architecture
from src.common_paths import get_model_path, get_tensorboard_logs_path

from tqdm import tqdm

batch_size = 128
project_id = "jigsaw"
version_id = "v0"
config = json.load(open("settings.json"))


columns_target = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df_train, df_dev, df_test, allowed_symbols, cle = load_train_data()


net = Architecture(class_cardinality=6, vocab_size=len(allowed_symbols)+3, name=project_id)

sess = start_tensorflow_session(device=str(config["device"]), memory_fraction=config["memory_fraction"])
fw = get_summary_writer(sess, get_tensorboard_logs_path(), project_id, version_id)
saver = TensorFlowSaver(get_model_path(project_id, version_id), max_to_keep=10)
sess.run(tf.global_variables_initializer())


losses_train = list()
min_loss = np.Inf
c=0

for epoch in range(10000):
    batcher_train = get_batcher(df_train, batch_size)
    pbar = tqdm(batcher_train, unit=" btch", total=df_train.shape[0]//batch_size, ncols=75)
    for i, (id_train, batch_train, target_train) in enumerate(pbar):
        _, s, l = sess.run([net.op.op, net.summaries.s_tr, net.losses.sigmoid_ce],
                        feed_dict={net.ph.comment_in: batch_train,
                                   net.ph.target: target_train,
                                   net.ph.is_train: True})

        fw.add_summary(s, c)
        pbar.set_description("[epoch: {0:04d} | loss: {1:.3f}]".format(epoch+1, np.mean(l)))
        c += 1

    losses_dev = []
    batcher_dev = get_batcher(df_dev, batch_size * 2)
    for j, (id_dev, batch_dev, target_dev) in enumerate(batcher_dev):
        loss_dev = sess.run(net.losses.sigmoid_ce, feed_dict={net.ph.comment_in: batch_dev,
                                                              net.ph.target: target_dev,
                                                              net.ph.is_train: False})
        losses_dev.append(np.mean(loss_dev))
    loss_dev = np.mean(losses_dev)
    s = sess.run(net.summaries.s_de, feed_dict={net.ph.loss_dev: loss_dev})
    fw.add_summary(s, epoch)
    pbar.set_description("[epoch: {0:04d} | loss: {1:.3f} | loss_dev: {2:.3f}]".format(epoch+1, np.mean(l), loss_dev))
    pbar.close()
    if loss_dev < min_loss:
        min_loss = loss_dev
        saver.save(sess, epoch)



