import os
import shutil

import tensorflow as tf


def get_tensorflow_configuration(device="0", memory_fraction=1):
    """
    Function for selecting the GPU to use and the amount of memory the process is allowed to use
    :param device: which device should be used (str)
    :param memory_fraction: which proportion of memory must be allocated (float)
    :return: config to be passed to the session (tf object)
    """
    device = str(device)

    if device:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        config.gpu_options.visible_device_list = device
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    return config


def start_tensorflow_session(device="0", memory_fraction=1):
    """
    Starts a tensorflow session taking care of what GPU device is going to be used and
    which is the fraction of memory that is going to be pre-allocated.
    :device: string with the device number (str)
    :memory_fraction: fraction of memory that is going to be pre-allocated in the specified
    device (float [0, 1])
    :return: configured tf.Session
    """
    return tf.Session(config=get_tensorflow_configuration(device=device, memory_fraction=memory_fraction))


def get_summary_writer(session, logs_path, project_id, version_id, remove_if_exists=True):
    """
    For Tensorboard reporting
    :param session: opened tensorflow session (tf.Session)
    :param logs_path: path where tensorboard is looking for logs (str)
    :param project_id: name of the project for reporting purposes (str)
    :param version_id: name of the version for reporting purposes (str)
    :param remove_if_exists: if True removes the log in case it exists (bool)
    :return summary_writer: the tensorboard writer
    """
    path = os.path.join(logs_path, "{}_{}".format(project_id, version_id))
    if os.path.exists(path) and remove_if_exists:
        shutil.rmtree(path)
    summary_writer = tf.summary.FileWriter(path, graph_def=session.graph_def)
    return summary_writer


class TensorFlowSaver:
    def __init__(self, path, max_to_keep=100):
        self.path = os.path.join(path, "model")
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def save(self, sess, step):
        self.saver.save(sess=sess, save_path=self.path, global_step=step)

    def load(self, sess, ckpt_file="latest"):
        path = os.path.split(self.path)[0]
        if ckpt_file == "latest":
            ckpt_path = tf.train.latest_checkpoint(path)
        else:
            ckpt_path = os.path.join(path, ckpt_file)
        self.saver.restore(sess, ckpt_path)
