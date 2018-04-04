from datetime import timedelta, datetime

import itertools

import os
from tensorflow.python.client import device_lib
import time
import tensorflow as tf

from config import FLAGS
from models.convnet.convnet import Model
from data_processing import get_train_valid_iterators, get_test_iterator

tf.logging.set_verbosity(tf.logging.INFO)


def training(iterator: tf.data.Iterator, handle: tf.placeholder, training_iterator, validation_iterator, dataset_size):
    tf_board_log_dir = os.path.join(FLAGS.log_dir, 'tf_board')
    num_iterations = 0
    start = time.time()
    image, label = iterator.get_next()
    model = Model(image, label, 4)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(tf_board_log_dir, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(tf_board_log_dir, 'test'))
        sess.run(tf.global_variables_initializer())
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        while True:
            epoch = int(FLAGS.batch_size * num_iterations / dataset_size[0])
            if num_iterations % 300 == 1:
                summary, acc = sess.run([merged, model.accuracy], feed_dict={handle: validation_handle})
                test_writer.add_summary(summary, num_iterations)
                tf.logging.info(f'Epoch: {epoch}, batch: {num_iterations}, test accuracy: {acc*100}%')
            if num_iterations % 100 == 1:  # Print train acc
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                try:
                    summary, acc = sess.run([merged, model.accuracy], feed_dict={handle: training_handle},
                                            options=run_options, run_metadata=run_metadata)
                except tf.errors.OutOfRangeError:
                    break
                train_writer.add_run_metadata(run_metadata, 'step%03d' % num_iterations)
                train_writer.add_summary(summary, num_iterations)
                tf.logging.info(
                    f"Epoch: {epoch}, batch: {num_iterations}, training accuracy: {acc * 100}%")
                saver.save(sess, os.path.join(FLAGS.log_dir, f'checkpoint_{datetime.now().isoformat()}.ckpt'))
            else:
                try:
                    summary, _ = sess.run([merged, model.optimize], feed_dict={handle: training_handle})
                    train_writer.add_summary(summary, num_iterations)
                except tf.errors.OutOfRangeError:
                    break

            num_iterations += 1
        validate(model, handle, validation_handle, sess)
    time_dif = time.time() - start
    tf.logging.info("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    train_writer.close()
    test_writer.close()


def validate(model, handle, validation_handle, sess, steps=None):
    i = 0
    total_acc = 0
    generator = itertools.count() if steps is None else range(steps)
    for i in generator:
        try:
            acc = sess.run(model.accuracy, feed_dict={handle: validation_handle})
            i += 1
            total_acc += acc
        except tf.errors.OutOfRangeError:
            break
    accuracy_valid = total_acc / i
    tf.logging.info("Average validation set accuracy is {:.2f}%".format(accuracy_valid * 100))
    return accuracy_valid


def test(test_iterator: tf.data.Iterator, dataset_size: int, checkpoint_path:str):
    num_iterations = 0
    start = time.time()
    with tf.Graph().as_default() as g:
        image, label = test_iterator.get_next()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
    # model = Model(image, label, 4)
    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        saver = tf.train.import_meta_graph(FLAGS.meta_path)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print('restored')
        print(sess.run('bias:0'))

def main():
    tf.logging.info(device_lib.list_local_devices())
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if not FLAGS.test:
        iterator, handle, training_iterator, validation_iterator, dataset_size = get_train_valid_iterators(
            FLAGS.data_path,
            FLAGS.test_size,
            FLAGS.epochs,
            FLAGS.batch_size,
            FLAGS.img_size,
            FLAGS.num_parallel,
            FLAGS.buffer_size)
        training(iterator, handle, training_iterator, validation_iterator, dataset_size)
    else:
        tf.logging.info(f"Running test on {FLAGS.test_set_path}")
        test_iterator, dataset_size = get_test_iterator(FLAGS.test_set_path, FLAGS.img_size, FLAGS.batch_size,
                                                        FLAGS.num_parallel)
        test(test_iterator, dataset_size, FLAGS.checkpoint_path)


if __name__ == '__main__':
    main()
