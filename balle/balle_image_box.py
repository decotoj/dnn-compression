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

AWS = 0  # Flag to denote training on AWS or not

# Tunable Hyperparameters
LMBDA = 0.01  # Lambda Parameter Controlling Rate vs Distortion Loss Function Weighting
LMBDA2 = 0.1  # Lambda Parameter  for portion inside bounding box Controlling Rate vs Distortion Loss Function Weighting
LEARNING_RATE = 1e-4  # Nominally

# Constants (Including Hyperparameters Treated as Constants)s
MODEL_DIRECTORY = 'model/google'  # Relative location where model files will be saved
TRAIN_DIRECTORY = 'images_BSD500/images_box/image/*.jpg'  # Training data relative directory
LABEL_DIRECTORY = 'images_BSD500/images_box/labels/*.txt'
NUM_STEPS = 100000  # Maximum number of steps to train (NOTE: 1,000,000 used by Balle)
PREPROCESS_THREADS = 16  # Number of CPU threads to use for parallel decoding of training images
NUM_FILTERS = 128  # Number of filters per layer in CNN
PATCHSIZE = 256  # Patch size
if AWS == 1:
    CHECKPOINT_SAVE = 60  # Number of seconds between checkpoint saves (60 on AWS, 1800 on local machine)
    LOG_STEPS = 1000  # Number of Steps Between Each Training Log Print (1000 on AWS, 10 on local machine)
    BATCH_SIZE = 32  # Batch size in training
else:
    CHECKPOINT_SAVE = 60  # Number of seconds between checkpoint saves (60 on AWS, 1800 on local machine)
    LOG_STEPS = 10  # Number of Steps Between Each Training Log Print (1000 on AWS, 10 on local machine)
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
    image = tf.image.decode_jpeg(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    # image = tf.image.resize_images(image, [640, 640], preserve_aspet_ratio = True)
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


# the below does return properly, if you run it
# it will print the contents of each .txt file
# however the contents are incldued in a sparse tensor
# so you'll see that most of the tensor's indicies have nothing in them
# when I try to convert into a dense tensor it doesn't work.
def load_labels(filename):
    string = tf.read_file(filename)
    # box =tf.constant(string, name='box')
    # x = np.loadtxt(filename.as_string())
    # xtensor = tf.constant(x,tf.string)
    string = tf.string_split([string])
    # string = tf.sparse_tensor_to_dense(string)
    tf.print(string)
    return string


# This was used in the CS230 pipeline webiste
# However, I dont know that its applicable here.
# It does return a dense tensor, which I think may be essential
def extract_char(token, default_value="<pad_char>"):
    # Split characters

    out = tf.string_split(token, delimiter=' ')
    # Convert to Dense tensor, filling with default value
    out = tf.sparse_tensor_to_dense(out, default_value=default_value)

    return out


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

    # Log Input Settings
    logFile = MODEL_DIRECTORY + '/' + 'Train_Log.txt'

    # Set Tensorflow Logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device('/cpu:0'):
        train_files = glob.glob(TRAIN_DIRECTORY)
        train_labels = glob.glob(LABEL_DIRECTORY)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)

        # NEW - The below seems to be one option to obtain information from
        # text files.  However, TF is extraordinarily difficult with respect to
        # being able to parse the text.  I've Googled this for hours, and
        # it's not explained as far as I can tell (it likely is of course)

        # label_dataset = tf.data.Dataset.from_tensor_slices(train_labels)

        # This was from the cs230 input pipeline website provided to us.
        # the only error it throws is that the read-in text files are of
        # a different size.  That is, some text files define multiple bounding
        # boxes.  I recommend we just use the first included bounding box;
        # this would give us 4 values for each text file then and there would
        # be no issue.
        label_dataset = tf.data.TextLineDataset(train_labels)
        # label_dataset = tf.data.TextLineDataset.from_tensor_slices(label_dataset)
        label_dataset = label_dataset.map(lambda token: tf.string_split([token]).values)
        label_dataset = label_dataset.map(lambda token: (token, extract_char(token)))

        # NEW - PLEASE REVIEW - we load images here
        # note that TF throws an error if any image is a different size
        # so we can either use the patch scheme of Balle, or we can resize
        # the images.  I'm not sure if the patch size would work, because
        # when we compute the MSE I dont know if TF first recombines all the patches
        # or if computes the MSE of each patch.  if its each patch then we would need
        # a function to check whether a patch includes a portion of a bounding box.
        # That said, if we resize the images it's unclear to me what size they should be
        # also we have to scale the bounding boxes to the new size somehow.

        train_dataset = train_dataset.map(load_image, num_parallel_calls=PREPROCESS_THREADS)
        train_dataset = train_dataset.map(
            lambda x: tf.random_crop(x, (PATCHSIZE, PATCHSIZE, 3)))

        # label_dataset = label_dataset.map(load_labels, num_parallel_calls=PREPROCESS_THREADS)

        # This combines the two datasets so they are coordinated.
        total_data = tf.data.Dataset.zip((train_dataset, label_dataset))
        total_data = total_data.shuffle(buffer_size=len(train_files)).repeat()

        # We prefetch some initial batches
        total_data = total_data.batch(BATCH_SIZE)
        total_data = total_data.prefetch(32)

        # train_labels = train_labels.batch(BATCH_SIZE)
        # train_labels = train_labels.prefetch(32)

    # Determine number of pixels and print input data info
    num_pixels = BATCH_SIZE * PATCHSIZE ** 2
    print('Num Train File', len(train_files))
    print('Num_Pix', num_pixels, BATCH_SIZE, PATCHSIZE)

    # Get Data - this includes labels and training images
    x = total_data.make_one_shot_iterator().get_next()

    # We then pass the training images in x[0] to our autoencoder
    y = analysis_transform(x[0], NUM_FILTERS)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde, NUM_FILTERS)

    # Total number of bits divided by number of pixels.
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x[0], x_tilde))
    train_mse *= 255 ** 2  # Multiply by 255^2 to correct for rescaling.

######################START TEST DECOTO############################

    #Grab the 4 Corners
    corners = [tf.string_to_number(x[1][1][1][2]), tf.string_to_number(x[1][1][1][3]), tf.string_to_number(x[1][1][1][4]), tf.string_to_number(x[1][1][1][5])]

    #Build a Mask of All 0,s of Proper Shape to Multiply With x[0] (Shape = 1,256,256,1)
    M = tf.zeros([1, x[0].get_shape()[1], x[0].get_shape()[1], 1])

    #START PENDING - WORK IN PROGRESS
    #Replace the 0's in M with 1's for all areas inside the bounding box
    indices = []
    values = []
    for i in range(0,10): #Replace 0 and 10 w/ the corner values
        for j in range(0,10): #Replace 0 and 10 w/ the corner values
            indices.append([0, i, j, 0]) #Indices of Values to Change
            values.append(1) #What to Change the Values at Indices To
    shape = M.get_shape()
    delta = tf.SparseTensor(indices, values, shape)
    delta = tf.cast(delta, tf.float32)
    M2 = M + tf.sparse_tensor_to_dense(delta)

    sums = [tf.reduce_sum(M), tf.reduce_sum(M2)] #Used to Print Later to Check This is Working (Sum of M = 0, Sum of M1 > 0)

    #END PENDING  - WORK IN PROGRESS

    #Mean Squared Error for the Box Portion Only
    train_mse_box = tf.reduce_mean(tf.multiply(tf.squared_difference(x[0], x_tilde), M2))
    train_mse_box *= 255 ** 2

    #Training Loss Including the Bounding Box as a separate loss component
    train_loss = LMBDA * train_mse + train_bpp + LMBDA2 * train_mse_box

###################END TEST DECOTO############################

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE * 10)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)

    tf.summary.image("original", quantize_image(x[0]))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    # Creates summary for the probability mass function (PMF) estimated in the bottleneck.
    entropy_bottleneck.visualize()

    hooks = [tf.train.StopAtStepHook(last_step=NUM_STEPS), tf.train.NanTensorHook(train_loss)]

    ep = 0
    epSub = 0
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
    with tf.train.MonitoredTrainingSession(scaffold=scaffold, hooks=hooks, checkpoint_dir=MODEL_DIRECTORY,
                                           save_checkpoint_secs=CHECKPOINT_SAVE,
                                           save_summaries_secs=CHECKPOINT_SAVE) as sess:
        while not sess.should_stop():
            sess.run(train_op)

            if epSub >= LOG_STEPS:
                epSub = 0
                ep += 1
            if epSub == 0:
                print(ep * LOG_STEPS + epSub, 'train loss', sess.run(train_loss))

######################START DECOTO EDITS######################################
                print('Corners', sess.run(corners))
                print('Sums M and M2', sess.run(sums))
######################END DECOTO EDITS######################################

                with open(logFile, 'a') as f:
                    f.write('step=' + str(ep * LOG_STEPS + epSub) + ',train_loss=' + str(
                        sess.run(train_loss)) + ',train_bpp=' + str(sess.run(train_bpp)) +
                            ',train_mse=' + str(sess.run(train_mse)) + '\n')
            epSub += 1

    print('TRAIN COMPLETED')


# Compress Image
def compress():
    print('Compress Image: ', args.input, NUM_FILTERS, MODEL_DIRECTORY, args.output)

    # Load input image and add batch dimension.
    x = load_image(args.input)
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
        with open(args.output, "wb") as f:
            f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
            f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
            f.write(string)


# Decompress Image
def decompress():
    # Read the shape information and compressed string from the binary file.
    with open(args.input, "rb") as f:
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
    op = save_image(args.output, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIRECTORY)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("command", choices=["train", "compress", "decompress"])
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "compress":
        compress()
    elif args.command == "decompress":
        decompress()