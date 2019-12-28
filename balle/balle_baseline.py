# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model of
Ballé, Laparra, Simoncelli (2017):
End-to-end optimized image compression
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>
"""

AWS = 0 #Flag to denote training on AWS or not

#Tunable Hyperparameters
LMBDA = 0.01  # Lambda Parameter Controlling Rate vs Distortion Loss Function Weighting
LEARNING_RATE = 1e-4 #Nominally

#Constants (Including Hyperparameters Treated as Constants)s
MODEL_DIRECTORY = 'model/balle_baseline'  # Relative location where model files will be saved
TRAIN_DIRECTORY = 'images_BSD500/images_train/*.png'  # Training data relative directory
NUM_STEPS = 100000 #Maximum number of steps to train (NOTE: 1,000,000 used by Balle)
PREPROCESS_THREADS = 16 #Number of CPU threads to use for parallel decoding of training images
NUM_FILTERS = 128 #Number of filters per layer in CNN
PATCHSIZE = 256 #Patch size
if AWS == 1:
  CHECKPOINT_SAVE = 60 #Number of seconds between checkpoint saves (60 on AWS, 1800 on local machine)
  LOG_STEPS = 1000 #Number of Steps Between Each Training Log Print (1000 on AWS, 10 on local machine)
  BATCH_SIZE = 32  # Batch size in training
else:
  CHECKPOINT_SAVE = 120  # Number of seconds between checkpoint saves (60 on AWS, 1800 on local machine)
  LOG_STEPS = 10 #Number of Steps Between Each Training Log Print (1000 on AWS, 10 on local machine)
  BATCH_SIZE = 2  # Batch size in training

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
import glob
import time

def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image

def quantize_image(image):
  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  return image

def save_image(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def train():
  """Trains the model."""

  #Log Input Settings
  logFile = MODEL_DIRECTORY + '/' + 'Train_Log.txt'

  #Set Tensorflow Logging
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files = glob.glob(TRAIN_DIRECTORY)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(load_image, num_parallel_calls=PREPROCESS_THREADS)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (PATCHSIZE, PATCHSIZE, 3)))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(32)

  #Determine number of pixels and print input data info
  num_pixels = BATCH_SIZE * PATCHSIZE ** 2
  print('Num Train File', len(train_files))
  print('Num_Pix', num_pixels, BATCH_SIZE, PATCHSIZE)

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Build autoencoder.
  y = analysis_transform(x, NUM_FILTERS)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, NUM_FILTERS)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  train_mse *= 255 ** 2 # Multiply by 255^2 to correct for rescaling.

  # The rate-distortion cost.
  train_loss = LMBDA * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*10)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the bottleneck.
  entropy_bottleneck.visualize()

  hooks = [tf.train.StopAtStepHook(last_step=NUM_STEPS),tf.train.NanTensorHook(train_loss)]

  ep = 0
  epSub = 0
  scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
  with tf.train.MonitoredTrainingSession(scaffold=scaffold, hooks=hooks, checkpoint_dir=MODEL_DIRECTORY,
      save_checkpoint_secs=CHECKPOINT_SAVE, save_summaries_secs=CHECKPOINT_SAVE) as sess:
    while not sess.should_stop():
      sess.run(train_op)

      if epSub >= LOG_STEPS:
        epSub = 0
        ep += 1
      if epSub == 0:
        print(ep*LOG_STEPS+epSub, 'train loss', sess.run(train_loss))
        with open(logFile, 'a') as f:
            f.write('step=' + str(ep*LOG_STEPS+epSub)  + ',train_loss=' + str(sess.run(train_loss)) + ',train_bpp=' + str(sess.run(train_bpp)) +
                    ',train_mse=' + str(sess.run(train_mse)) + '\n')
      epSub += 1

  print('TRAIN COMPLETED')

#Compress Image
def compress(input, output):

  tf.reset_default_graph()

  # Load input image and add batch dimension.
  x = load_image(input)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(x, NUM_FILTERS)
  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIRECTORY)
    tf.train.Saver().restore(sess, save_path=latest)
    string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

    # Write a binary file with the shape information and the compressed string.
    with open(output, "wb") as f:
      f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(string)


    sess.close()

#Decompress Image
def decompress(input, output):

  tf.reset_default_graph()

  # Read the shape information and compressed string from the binary file.
  with open(input, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [NUM_FILTERS]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(strings, y_shape, channels=NUM_FILTERS)
  x_hat = synthesis_transform(y_hat, NUM_FILTERS)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(output, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIRECTORY)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)

    sess.close()

if __name__ == "__main__":
  train()
