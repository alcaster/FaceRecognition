from datetime import timedelta, datetime
from tensorflow.python.client import device_lib
import time
import tensorflow as tf

from config import cfg
from models.convnet.convnet import Model
from data_processing import get_train_n_test_datasets

tf.logging.set_verbosity(tf.logging.INFO)


def training(iterator: tf.data.Iterator, training_init_op, validation_init_op, log_dir):
    image, label = iterator.get_next()
    num_iterations = 0
    start = time.time()
    model = Model(image, label, 4)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(f'{log_dir}/tf_board',
                                             sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(training_init_op)
        while True:
            try:
                summary, _ = sess.run([merged, model.optimize])
                train_writer.add_summary(summary, num_iterations)
            except tf.errors.OutOfRangeError:
                break
            if num_iterations % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, acc = sess.run([merged, model.accuracy], options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % num_iterations)
                train_writer.add_summary(summary, num_iterations)
                tf.logging.info(f"Batch: {num_iterations}, training accuracy: {acc * 100}%")
                saver.save(sess, f'{log_dir}/checkpoint_{datetime.now().isoformat()}.ckpt')
            num_iterations += 1
        validate(model, validation_init_op, sess)
    time_dif = time.time() - start
    tf.logging.info("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    train_writer.close()


def validate(model, validation_init_op, sess):
    sess.run(validation_init_op)
    i = 0
    total_acc = 0
    while True:
        try:
            acc = sess.run(model.accuracy)
            i += 1
            total_acc += acc
        except tf.errors.OutOfRangeError:
            break
    tf.logging.info("Average validation set accuracy is {:.2f}%".format((total_acc / i) * 100))


def main():
    tf.logging.info(device_lib.list_local_devices())
    if tf.gfile.Exists(cfg.log_dir):
        tf.gfile.DeleteRecursively(cfg.log_dir)
    tf.gfile.MakeDirs(cfg.log_dir)
    iterator, training_init_op, validation_init_op = get_train_n_test_datasets(cfg.data_path, cfg.test_size, cfg.epochs,
                                                                               cfg.batch_size,
                                                                               cfg.img_size, cfg.num_parallel,
                                                                               cfg.buffer_size)
    if cfg.train:
        training(iterator, training_init_op, validation_init_op, cfg.log_dir)


if __name__ == '__main__':
    main()
