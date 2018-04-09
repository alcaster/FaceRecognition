import tensorflow as tf

from models.convnet.layers import new_conv_layer, new_fc_layer, flatten_layer
from utils.wrappers import define_scope


class Model:

    def __init__(self, image, label, num_classes, lr=1e-3):
        self.image = image
        # tf.summary.image('preprocessed_image', image, max_outputs) # Stored too many images, producing very big tensorboard save files.
        self.label = label
        self.NUM_CLASSES = num_classes
        self.CHANNELS = 3
        self.lr = lr
        self.prediction
        self.optimize
        self.accuracy

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        # Conv1
        filter_size1 = 5
        num_filters1 = 16
        # Conv2.
        filter_size2 = 5
        num_filters2 = 36
        # Fully-connected.
        fc_size = 128

        layer_conv1, weights_conv1 = \
            new_conv_layer(input=self.image,
                           num_input_channels=self.CHANNELS,
                           filter_size=filter_size1,
                           num_filters=num_filters1,
                           use_pooling=True, scope="Conv1")
        layer_conv2, weights_conv2 = \
            new_conv_layer(input=layer_conv1,
                           num_input_channels=num_filters1,
                           filter_size=filter_size2,
                           num_filters=num_filters2,
                           use_pooling=True, scope="Conv2")
        layer_flat, num_features = flatten_layer(layer_conv2)
        layer_fc1 = new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=fc_size,
                                 use_relu=True, scope="Fully1")
        layer_fc2 = new_fc_layer(input=layer_fc1,
                                 num_inputs=fc_size,
                                 num_outputs=self.NUM_CLASSES,
                                 use_relu=False, scope="Fully2")
        return layer_fc2

    @define_scope
    def optimize(self):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction,
                                                                       labels=self.label)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        return optimizer.minimize(loss)

    @define_scope
    def accuracy(self):
        y_pred = tf.nn.softmax(self.prediction)
        y_pred_cls = tf.argmax(y_pred, axis=1)
        ground_truth = tf.argmax(self.label, 1)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(y_pred_cls, ground_truth)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
