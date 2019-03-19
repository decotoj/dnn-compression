#Implementation of Adapted Li Algorithm for Image Compression

#Algorithm based on paper available here: https://arxiv.org/abs/1703.10553

#Jake Decoto and Brian Graham
#3/15/2019

#ACKNOWLEDGMENT: Some structural elements such as code for compressing and decompressing
#images given a pre-trained model, entropy model, and template for calling tensorflow
#were borrowed from https://github.com/tensorflow/compression
#Such elements come with the below restriction.

# ==============================================================================
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

compress_type = 'NA' #set to 'map' for compression/decompression of P

AWS = 0 #Flag to denote training on AWS or not

#Tunable Hyperparameters
LMBDA = 0.01  #Relative Importance of Distortion vs Rate Loss (Higher Value Weights Distortion Loss More Heavily)
LEARNING_RATE = 1e-4 #Nominally 1e-4
MASK_THRESHOLD = 0.5 #Nominally 0.1
RATE_LOSS_THRESHOLD = 20000 #Point Below Which Rate Loss is Set to Zero (Used to Govern Approx Compression Ratio)

#Constants (Including Hyperparameters Treated as Constants)
MODEL_DIRECTORY = 'model/li_004'  # Relative location where model files will be saved
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
  CHECKPOINT_SAVE = 60  # Number of seconds between checkpoint saves (60 on AWS, 1800 on local machine)
  LOG_STEPS = 10 #Number of Steps Between Each Training Log Print (1000 on AWS, 10 on local machine)
  BATCH_SIZE = 1  # Batch size in training

#Import Libraries
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

#Quantitize Image
def quantize_image(image):
  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  return image

#Save Image as PNG File
def save_image(filename, image):
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

#Encoder
def encoder_li(tensor):

    with tf.variable_scope("analysis"):
        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                NUM_FILTERS, (9, 9), corr=True, strides_down=4, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                NUM_FILTERS, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                NUM_FILTERS, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=False, activation=tfc.GDN())
            tensor = layer(tensor)
            tensor2 = tensor

        with tf.variable_scope("layer_3"):
            layer = tfc.SignalConv2D(
              NUM_FILTERS, (1, 1), corr=True, strides_down=1, padding="same_zeros",
              use_bias=False, activation=None)
            tensor = layer(tensor)

        return tensor, tensor2

#Decoder
def decoder_li(tensor):

  with tf.variable_scope("synthesis"):

    with tf.variable_scope("layer_0"):
         layer = tfc.SignalConv2D(
             NUM_FILTERS, (1, 1), corr=True, strides_down=1, padding="same_zeros",
             use_bias=True, activation=tfc.GDN(inverse=True))
         tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          NUM_FILTERS, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          NUM_FILTERS, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

  return tensor

#Binarizer Defined In Three Functions to Allow for Non-Default Backpropagation Behavior
#Follows Custom Backprop Advice Here: https://twitter.com/ericjang11/status/932073259721359363?lang=en
def f(tensor):
    return tf.round(tf.nn.sigmoid(tensor))

def g(tensor):
    return tf.clip_by_value(tensor, 0, 1)

def binarizer(tensor):
    with tf.variable_scope("binarizer"):
        return tf.math.subtract(tf.math.add(f(tensor), g(tensor)), tf.stop_gradient(g(tensor)))

#Importance Map
def importance_map(tensor):

    with tf.variable_scope("importance"):

        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                NUM_FILTERS, (1, 1), corr=True, strides_down=1, padding="same_zeros",
                use_bias=True, activation=tf.nn.sigmoid)
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                NUM_FILTERS, (1, 1), corr=True, strides_down=1, padding="same_zeros",
                use_bias=True, activation=tf.nn.sigmoid)
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                NUM_FILTERS, (1, 1), corr=True, strides_down=1, padding="same_zeros",
                use_bias=True, activation=tf.nn.sigmoid)
            tensor = layer(tensor)

        return tensor

#Mask Defined In Three Functions to Allow for Non-Default Backpropagation Behavior
#Follows Custom Backprop Advice Here: https://twitter.com/ericjang11/status/932073259721359363?lang=en
def fm(tensor):
    return tf.ceil(tf.math.subtract(tensor, MASK_THRESHOLD))

def gm(tensor):
    return tf.clip_by_value(tensor, 0, 1)

def gen_mask(tensor):
    with tf.variable_scope("binarizer"):
        return tf.math.subtract(tf.math.add(fm(tensor), gm(tensor)), tf.stop_gradient(gm(tensor)))

#Rate Loss Function
def rate_loss(tensor):
    return tf.clip_by_value(tf.reduce_sum(tensor)-RATE_LOSS_THRESHOLD, 0, 10000000)

#Train Model
def train():

  # #Log Input Settings
  logFile = MODEL_DIRECTORY + '/' 'Train_Log.txt'

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

  ###########################Li Algrithm Start#################################

  # Build autoencoder & decoder
  E, fx = encoder_li(x)

  P = importance_map(fx)
  M = gen_mask(P)

  B = binarizer(E)

  bc = tf.multiply(E,M) #NOTE: Skipping 'B' and Using "E' instead seemed to work better

  entropy_bottleneck = tfc.EntropyBottleneck()
  bc_tilde, likelihoods = entropy_bottleneck(bc, training=True)

  x_tilde = decoder_li(bc_tilde)

  print('x', x)
  print('E', E)
  print('fx', fx)
  print('x_tilde', x_tilde)
  print('map', P)
  print('B', B)
  print('M', M)
  print('bc', bc)

  #Rate Loss
  rateLoss = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  train_mse *= 255 ** 2 # Multiply by 255^2 to correct for rescaling.

  # The rate-distortion cost.
  train_loss = LMBDA * train_mse + rateLoss #TEST1234

  ###########################Li Algrithm End#################################

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*10)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  # # #Check Values################
  # Pstats = [tf.math.reduce_min(P), tf.math.reduce_max(P), tf.reduce_sum(P)]
  # Estats = [tf.math.reduce_min(E), tf.math.reduce_max(E), tf.reduce_sum(E), tf.size(E)]
  # Mstats = [tf.math.reduce_min(M), tf.math.reduce_max(M),tf.reduce_sum(M), tf.size(M)]
  # Bstats = [tf.math.reduce_min(B), tf.math.reduce_max(B),tf.reduce_sum(B), tf.size(B)]
  # BCstats = [tf.math.reduce_min(bc), tf.math.reduce_max(bc), tf.reduce_sum(bc), tf.size(bc)]
  # XTstats = [tf.math.reduce_min(x_tilde), tf.math.reduce_max(x_tilde), tf.reduce_sum(x_tilde)]
  # # # ##############################

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", rateLoss)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the bottleneck.
  entropy_bottleneck.visualize()

  hooks = [tf.train.StopAtStepHook(last_step=NUM_STEPS),tf.train.NanTensorHook(train_loss),]

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
        print(ep*LOG_STEPS+epSub, 'TRAIN/DIST/RATE LOSS:', sess.run(train_loss), sess.run(train_mse), sess.run(rateLoss))
        # print('    Estats', sess.run(Estats))
        # print('    Pstats', sess.run(Pstats))
        # print('    Mstats', sess.run(Mstats))
        # print('    Bstats', sess.run(Bstats))
        # print('    BCstats', sess.run(BCstats))
        # print('    XTstats', sess.run(XTstats))
        with open(logFile, 'a') as f:
            f.write('step=' + str(ep*LOG_STEPS+epSub)  + ',train_loss=' + str(sess.run(train_loss)) + ',rateLoss=' + str(sess.run(rateLoss)) +
                    ',distortionLoss=' + str(sess.run(train_mse)) + '\n')
      epSub += 1

#Compress Image Using Pre-Trained Model
def compress(input, output):

  tf.reset_default_graph()

  # Load input image and add batch dimension.
  x = load_image(input)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  E, fx = encoder_li(x)

  P = importance_map(fx)
  M = gen_mask(P)

  B = binarizer(E)

  bc = tf.multiply(E,M)

  if compress_type == 'map':
      y = P
      print('MAP')
  else:
      y = bc

  #Check Values
  Pstats = [tf.math.reduce_min(P), tf.math.reduce_max(P), tf.reduce_sum(P)]
  Mstats = [tf.math.reduce_min(M), tf.math.reduce_max(M), tf.reduce_sum(M)]

  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  if compress_type == 'map':
    string = tf.squeeze(P, axis=0)

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIRECTORY)
    tf.train.Saver().restore(sess, save_path=latest)
    string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

    print('P stats', sess.run(Pstats))
    print('M stats', sess.run(Mstats))

    # Write a binary file with the shape information and the compressed string.
    with open(output, "wb") as f:
      f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(string)

# Decompress Image Using Pre-Trained Model
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
    x_hat = decoder_li(y_hat)

    # Remove batch dimension & crop any extraneous padding on the bottom or right boundaries.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = save_image(output, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIRECTORY)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op)

if __name__ == "__main__":

  if args.command == "train":
    train()