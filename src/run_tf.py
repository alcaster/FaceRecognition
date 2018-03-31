from data_processing import get_train_n_test_datasets
from tensorflow.python.client import device_lib
import time
import tensorflow as tf

from config import cfg

tf.logging.set_verbosity(tf.logging.INFO)


def training(iterator, training_init_op, validation_init_op):
    init_op = tf.global_variables_initializer()

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(training_init_op)
        x, y = iterator.get_next()



def main():
    print(device_lib.list_local_devices())
    iterator, training_init_op, validation_init_op = get_train_n_test_datasets(cfg.data_path, cfg.test_size, cfg.epochs,
                                                                               cfg.batch_size,
                                                                               cfg.img_size, cfg.num_parallel,cfg.buffer_size)
    if cfg.train:
        training(iterator, training_init_op, validation_init_op)


if __name__ == '__main__':
    main()
