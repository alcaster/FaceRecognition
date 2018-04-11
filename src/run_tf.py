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
    num_iterations = 0
    start = time.time()
    image, label = iterator.get_next(name="dataset_input")
    model = Model(image, label, 4)
    init = tf.group(tf.global_variables_initializer())

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'test'))
        sess.run(init)
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        while True:
            epoch = int(FLAGS.batch_size * num_iterations / dataset_size[0])
            if num_iterations % 300 == 1:
                summary, (acc, precision, recall) = sess.run([merged, model.metrics], feed_dict={handle: validation_handle})
                test_writer.add_summary(summary, num_iterations)
                tf.logging.info(
                    f'Epoch: {epoch}, batch: {num_iterations}, test - accuracy: {acc*100}%, precision: {precision*100}%, recall: {recall*100}%')
            if num_iterations % 100 == 1:  # Print train acc
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                try:
                    summary, (acc, precision, recall) = sess.run([merged, model.metrics],
                                                                 feed_dict={handle: training_handle},
                                                                 options=run_options, run_metadata=run_metadata)
                except tf.errors.OutOfRangeError:
                    break
                train_writer.add_run_metadata(run_metadata, 'step%03d' % num_iterations)
                train_writer.add_summary(summary, num_iterations)
                tf.logging.info(
                    f"Epoch: {epoch}, batch: {num_iterations}, training accuracy: {acc * 100}%, precision: {precision*100}%, recall: {recall*100}%")
                saver.save(sess, os.path.join(FLAGS.log_dir, 'model'), global_step=num_iterations)
            else:
                try:
                    _ , _ = sess.run([merged, model.optimize], feed_dict={handle: training_handle})
                    # train_writer.add_summary(summary, num_iterations)
                except tf.errors.OutOfRangeError:
                    break
            num_iterations += 1
        validate(model, handle, validation_handle, sess)
    time_dif = time.time() - start
    tf.logging.info("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    train_writer.close()
    test_writer.close()


def validate(model, handle, validation_handle, sess, steps=None):
    """
    Batches always have the same size so it is possible to take average of each metric.
    """
    i = 0
    total_acc, total_prec, total_rec = 0, 0, 0
    generator = itertools.count() if steps is None else range(steps)
    for _ in generator:
        try:
            acc, prec, rec = sess.run(model.metrics, feed_dict={handle: validation_handle})
            i += 1
            total_acc += acc
            total_prec += prec
            total_rec += rec
        except tf.errors.OutOfRangeError:
            break
    accuracy_valid = total_acc / i
    precision_valid = total_prec / i
    recall_valid = total_rec / i

    tf.logging.info(
        "Average validation set accuracy:{:.2f}%, accuracy:{:.2f}%, accuracy:{:.2f}%".format(accuracy_valid * 100,
                                                                                             precision_valid * 100,
                                                                                             recall_valid * 100))
    return accuracy_valid


def test(test_iterator: tf.data.Iterator, dataset_size: int, checkpoint_path: str):
    start = time.time()
    image, label = test_iterator.get_next()
    model = Model(image, label, 4)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        tf.logging.info(f"Successfully loaded model from checkpoint:{checkpoint_path}")
        i = 0
        total_acc, total_prec, total_rec = 0, 0, 0
        while True:
            try:
                acc, prec, rec = sess.run(model.metrics)
                print(acc)  # lots of 0 later hmmmm
                i += 1
                total_acc += acc
                total_prec += prec
                total_rec += rec
            except tf.errors.OutOfRangeError:
                break

    tf.logging.info(f"Dataset size {dataset_size}; Iterations {i}; Batch size {FLAGS.batch_size}")
    accuracy_valid = total_acc / i
    precision_valid = total_prec / i
    recall_valid = total_rec / i
    time_dif = time.time() - start
    tf.logging.info(
        "Average validation set accuracy:{:.2f}%, precision:{:.2f}%, recall:{:.2f}%".format(accuracy_valid * 100,
                                                                                            precision_valid * 100,
                                                                                            recall_valid * 100))
    tf.logging.info("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def main():
    tf.logging.info(device_lib.list_local_devices())
    if FLAGS.test:
        tf.logging.info(f"Running test on {FLAGS.test_set_path}")
        test_iterator, dataset_size = get_test_iterator(FLAGS.test_set_path, FLAGS.img_size, FLAGS.batch_size,
                                                        FLAGS.num_parallel)
        tf.logging.info(f"Test dataset size {dataset_size}")
        test(test_iterator, dataset_size, FLAGS.checkpoint_path)
    else:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
        iterator, handle, training_iterator, validation_iterator, dataset_size = get_train_valid_iterators(
            FLAGS.data_path,
            FLAGS.test_size,
            FLAGS.epochs,
            FLAGS.batch_size,
            FLAGS.img_size,
            FLAGS.num_parallel,
            FLAGS.buffer_size)
        training(iterator, handle, training_iterator, validation_iterator, dataset_size)


if __name__ == '__main__':
    main()
