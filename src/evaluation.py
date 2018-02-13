import json
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import roc_auc_score
from src.general_utilities import flatten
from src.architecture import Architecture
from src.common_paths import get_model_path, get_tensorboard_logs_path, get_observers_path
from src.data_tools import load_train_data, get_batcher
from src.tensorflow_utilities import start_tensorflow_session, get_summary_writer, TensorFlowSaver
from src.architecture import Architecture
from src.common_paths import get_model_path, get_tensorboard_logs_path

run_number = 3
project_id = "jigsaw"
version_id = "v1"
observer_path = os.path.join(get_observers_path(), str(run_number))
locals().update(json.load(open(os.path.join(observer_path, "config.json"))))
batch_size = 256
config = json.load(open("settings.json"))

df_train, df_dev, df_test, allowed_symbols, cle = load_train_data()

net = Architecture(class_cardinality=6, vocab_size=len(allowed_symbols) + 3, embedding_size=embedding_size,
                   n_recurrent_units=n_recurrent_units, learning_rate=learning_rate, name=project_id)

sess = start_tensorflow_session(device=str(config["device"]), memory_fraction=config["memory_fraction"])
saver = TensorFlowSaver(get_model_path(project_id, version_id))
saver.load(sess)

batcher_scoring = get_batcher(df_test, batch_size, train=True)
pbar = tqdm(batcher_scoring, unit=" btch", total=df_test.shape[0] // batch_size, ncols=75)
targets=[]
outputs=[]
ids=[]

for i, (id, batch, target) in enumerate(pbar):
    output = sess.run(net.core_model.output, feed_dict={net.ph.comment_in: batch,
                                                         net.ph.is_train: False,
                                                         net.ph.keep_prob: 1})
    ids.append(id)
    outputs.append(output)
    targets.append(target)

pbar.close()

ids = flatten(ids)
outputs = np.concatenate(outputs, axis=0)
targets = np.concatenate(targets, axis=0)

aucs = []
for i in range(targets.shape[1]):
    aucs.append(roc_auc_score(targets[:,i], outputs[:,i]))

print(aucs)