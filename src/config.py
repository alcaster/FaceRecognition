import tensorflow as tf

flags = tf.app.flags
############################
#   training setting    #
############################
flags.DEFINE_bool('train', True, 'Is training?')
flags.DEFINE_integer('batch_size', '64', 'Batch size.')
flags.DEFINE_float('test_size', '0.2', 'Percentage for test set. F.ex 0.2')
flags.DEFINE_integer('epochs', '50', 'Number of epochs.')

############################
#   Image preprocessing    #
############################
flags.DEFINE_integer('img_size', '220', 'Size of each image.')
flags.DEFINE_integer('num_parallel', '8', 'Amount of paralleling processed pictures.')
flags.DEFINE_integer('buffer_size', '100', 'Buffer size for image processing.')

############################
#   environment setting    #
############################
flags.DEFINE_string('data_path', '../data/train',
                    'The path to the train dataset.\n Each class should be in separate directory.')
flags.DEFINE_string('log_dir', '../out/1', 'Directory where saved checkpoints will be stored.')
cfg = tf.app.flags.FLAGS
