# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)

# Functions to tell TensorFlow how to read a single image from input file - ours first, example repo commented below
def read_and_decode(filename):
    # convert filenames to a queue for an input pipeline.
    #filenameQ = tf.train.string_input_producer([filename],num_epochs=None)
 
    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example 
    key, fullExample = recordReader.read(filename)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels':  tf.FixedLenFeature([], tf.int64),            
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })


    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    #with tf.name_scope('decode_jpeg',[image_buffer], None):
        # decode
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    
        # and convert to single precision data type
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel. 
    # the "1-\cdot" part inverts the image, so that the background is black.

    image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[256*256])
    image = tf.cast(image, tf.float32) * (1. / 255)

    # re-define label as a "one-hot" vector 
    # it will be one of [1,0,...,0], ..., [0,...,0,1] 

    #label=tf.stack(tf.one_hot(label, 7))

    return image, label


#def read_and_decode(filename_queue):
  #reader = tf.TFRecordReader()
  #_, serialized_example = reader.read(filename_queue)

  #features = tf.parse_single_example(
      #serialized_example,
      #features={
          #'image_raw': tf.FixedLenFeature([], tf.string),
          #'label': tf.FixedLenFeature([], tf.int64),
      #})

  #image = tf.decode_raw(features['image_raw'], tf.uint8)
  #image.set_shape([256*256])
  #image = tf.cast(image, tf.float32) * (1. / 255)
  #label = tf.cast(features['label'], tf.int32)

  #return image, label


def input_fn(filename, batch_size=100, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs)

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
      [image, label], batch_size=batch_size,
      capacity=1000 + 3 * batch_size)

    return {'image': images}, labels


def get_input_fn(filename, num_epochs=None, batch_size=100):
    return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
    
    
    #COPY/PASTE inference fnc from tf_workbench/blob/master/alexnet_redo/trainer/model.py modified for grayscale
    
    """Bild model up to where it may be used for inference
      Returns:
        softmax_linear: output tnesor with the computed logits
      """
    #input_layer = tf.reshape(images, [-1, 256, 256, 1])
    input_layer = tf.reshape(features['image'], [-1, 256, 256, 1])
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[11, 11],
            stride=4,
            padding="VALID",
            scope='conv1'
            )
    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                  kernel_size=[3, 3],
                                  stride=2,
                                  scope='pool1'
                                  )
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
              inputs=pool1,
              filters=192,
              kernel_size=[5, 5],
              scope='conv2'
              )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                  kernel_size=[3, 3],
                                  stride=2,
                                  scope='pool2'
                                  )
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        scope='conv3'
        )
    conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=384,
            kernel_size=[3, 3],
            scope='conv4'
            )
    conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=256,
            kernel_size=[3, 3],
            scope='conv5'
            )
    pool5 = tf.layers.max_pooling2d(inputs=conv5, 
                                  kernel_size=[3, 3],
                                  stride=2,
                                  scope='pool5')
    
    flattened = tf.layers.flatten(inputs=pool5, scope='flat')

    fc6 = tf.layers.dense(inputs=flattened,
                        units=4096,
                        scope='fc6')

    dropout6 = tf.layers.dropout(inputs=fc6,
                                 rate=0.5,
                                 training=(mode == learn.ModeKeys.TRAIN),
                                 scope='dropout6')

    fc7 = tf.layers.dense(inputs=dropout6,
                         units=4096,
                         scope='fc7')

    dropout7 = tf.layers.dropout(inputs=fc7,
                                 rate=0.5,
                                 training=(mode == learn.ModeKeys.TRAIN),
                                 scope='dropout7')

    fc8 = tf.layers.dense(inputs=dropout7,
                         units=4096,
                         #activation=tf.nn.relu,
                         scope='fc8')
    dropout8 = tf.layers.dropout(inputs=fc8,
                                 rate=0.5,
                                 training=(mode == learn.ModeKeys.TRAIN),
                                 scope='dropout8')

    logits = tf.layers.dense(inputs=dropout8,
                                     units=7,
                                     scope='logits')
    
    # associate the "label" and "image" objects with the corresponding features read from 
    # a single example in the training data file
    #image, label = read_and_decode("data/train-00000-of-00001")
    
    # and similarly for the validation data
    #vimage, vlabel = read_and_decode("data/validation-00000-of-00001")
    
    # associate the "label_batch" and "image_batch" objects with a randomly selected batch---
    # of labels and images respectively
    #imageBatch, labelBatch = tf.train.shuffle_batch(
        #[image, label], batch_size=100,
        #capacity=2000,
        #min_after_dequeue=0)
    
    # and similarly for the validation data 
    #vimageBatch, vlabelBatch = tf.train.shuffle_batch(
        #[vimage, vlabel], batch_size=100,
        #capacity=2000,
        #min_after_dequeue=0)    
    
    
  # Input Layer
    #input_layer = tf.reshape(imageBatch, [-1, 256, 256, 1])
    #input_layer = tf.reshape(features['image'], [-1, 256, 256, 1])

  # Convolutional Layer #1
    #conv1 = tf.layers.conv2d(
      #inputs=input_layer,
      #filters=96,
      #kernel_size=[11, 11],
      #stride = 4,
      #padding="valid",
      #activation=tf.nn.relu)

  # Pooling Layer #1
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
    #conv2 = tf.layers.conv2d(
      #inputs=pool1,
      #filters=64,
      #kernel_size=[5, 5],
      #padding="same",
      #activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
    #pool2_flat = tf.reshape(pool2, [-1, 262144])
    #dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    #dropout = tf.layers.dropout(
      #inputs=dense, rate=0.4, training=(mode == learn.ModeKeys.TRAIN))

  # Logits Layer
    #logits = tf.layers.dense(inputs=dropout, units=7)

    loss = None
    train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=7)
        loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=0.001, optimizer="Adam")

  # Generate Predictions
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, loss=loss, train_op=train_op,
                                 predictions=predictions)


def build_estimator(model_dir):
    return learn.Estimator(
           model_fn=_cnn_model_fn,
           model_dir=model_dir,
           config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def get_eval_metrics():
    return {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                       prediction_key="classes")
  }


def serving_input_fn():
    feature_placeholders = {'image': tf.placeholder(tf.float32, [None, 256*256])}
    features = {
    key: tensor
    for key, tensor in feature_placeholders.items()
  }    
    return learn.utils.input_fn_utils.InputFnOps(
    features,
    None,
    feature_placeholders
  )
