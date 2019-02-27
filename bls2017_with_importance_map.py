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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
import argparse
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

#Load Image File
def load_image(filename):
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

###############START###############
def importance_map(tensor, num_filters):

    with tf.variable_scope("importance"):
        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                num_filters, (5, 5), corr=True, strides_down=4, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        # with tf.variable_scope("layer_2"):
        #     layer = tfc.SignalConv2D(
        #         num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
        #         use_bias=False, activation=None)
        #     tensor = layer(tensor)

        return tf.sigmoid(tensor)
###############END###############

def train():
  """Trains the model."""

  #Log Input Settings
  logFile = str(round(time.time(),0)) + '_train_log.txt'
  with open(logFile, 'w') as f:
      for key,value in vars(args).items():
        f.write(key + '=' + str(value) + '\n')
      f.write('\n')

  #Set Tensorflow Logging
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files = glob.glob(args.train_glob)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(load_image, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  #Determine number of pixels and print input data info
  num_pixels = args.batchsize * args.patchsize ** 2
  print('Num Train File', len(train_files))
  print('Num_Pix', num_pixels, args.batchsize, args.patchsize)

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()
  print('x', x)

  # Build autoencoder.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, args.num_filters)

  map = importance_map(x, args.num_filters) #TEST1234

  print('likelihoods', likelihoods)
  print('x_tilde', x_tilde)
  print('y_tilde', y_tilde)
  print('map', map)

  # Total number of bits divided by number of pixels.
  #train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  train_bpp = tf.reduce_sum(tf.log(tf.multiply(likelihoods,map))) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  train_mse *= 255 ** 2 # Multiply by 255^2 to correct for rescaling.

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the bottleneck.
  entropy_bottleneck.visualize()

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]

  ep = 0
  epSub = 0
  nm = 10 #Number of Steps Between Each Log Print
  with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=600, save_summaries_secs=600) as sess:
    while not sess.should_stop():
      sess.run(train_op)

      if epSub >= nm:
        epSub = 0
        ep += 1
      if epSub == 0:
        print(ep*nm+epSub, 'train loss', sess.run(train_loss))
        with open(logFile, 'a') as f:
            f.write('step=' + str(ep*nm+epSub)  + ',train_loss=' + str(sess.run(train_loss)) + ',train_bpp=' + str(sess.run(train_bpp)) +
                    ',train_mse=' + str(sess.run(train_mse)) + '\n')
      epSub += 1

  print('TRAINING COMPLETED')

#Compress Image Using Pre-Trained Model
def compress():

  print('Compress Image: ', args.input, args.num_filters, args.checkpoint_dir, args.output)

  # Load input image and add batch dimension.
  x = load_image(args.input)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

    # Write a binary file with the shape information and the compressed string.
    with open(args.output, "wb") as f:
      f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(string)

#Decompress Image Using Pre-Trained Model
def decompress():

  # Read the shape information and compressed string from the binary file.
  with open(args.input, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [args.num_filters]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(
      strings, y_shape, channels=args.num_filters)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  # Remove batch dimension, and crop away any extraneous padding on the bottom or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(args.output, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress"],
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options.")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--verbose", "-v", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="model",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--train_glob", default="images_train/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      #"--batchsize", type=int, default=1,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      #"--patchsize", type=int, default=64,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  args = parser.parse_args()

  if args.command == "train":
    train()
  elif args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress()
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress()