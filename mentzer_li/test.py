from Main import *
import argparse
import glob
import time
import numpy as np
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as image
import keras.backend as K

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

  def msssmm_loss(x_input, x_decode):
  	return tf.image.ssim_multiscale (x_input, x_decode)

  def center_loss(centers):
  	return tf.nn.l2_loss(centers)
  

#def customLoss():
	#return K.sum(K.log(yTrue) - K.log(yPred))


  #Set Tensorflow Logging
def train():
	
	#MAY NEED THE BELOW
	#tf_sess = tf.Session()
	#K.set_session(tf_sess)
	""""
	logFile = str(round(time.time(),0)) + '_train_log.txt'
	with open(logFile, 'w') as f:
		for key,value in vars(args).items():
			f.write(key + '=' + str(value) + '\n')
		f.write('\n')
	tf.logging.set_verbosity(tf.logging.INFO)"""

	train_files = str(glob.glob(args.train_glob))

	test_datagen=image.ImageDataGenerator(rescale=1./255)
	train_generator = test_datagen.flow_from_directory(directory = 'fake_train', target_size=(640, 640), batch_size=args.batchsize, class_mode = 'input')


	"""with tf.device('/cpu:0'):
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
	x = train_dataset.make_one_shot_iterator().get_next()"""

					  # Build autoencoder.
					  #y = analysis_transform(x, args.num_filters)
					  #entropy_bottleneck = tfc.EntropyBottleneck()
					  #y_tilde, likelihoods = entropy_bottleneck(y, training=True)
					  #x_tilde = synthesis_transform(y_tilde, args.num_filters)

	  # Total number of bits divided by number of pixels.
	  #train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

	  # Mean squared error across pixels.
	  #train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
	  #train_mse *= 255 ** 2 # Multiply by 255^2 to correct for rescaling.

	  # The rate-distortion cost.
	  #train_loss = args.lmbda * train_mse + train_bpp

	  # Minimize loss and auxiliary loss, and execute update op.
	#step = tf.train.create_global_step()
	#main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
	#main_step = main_optimizer.minimize(train_loss, global_step=step)

  #aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  #aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  #train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  #tf.summary.scalar("loss", train_loss)
  #tf.summary.scalar("bpp", train_bpp)
  #tf.summary.scalar("mse", train_mse)
 
	#tf.summary.image("original", quantize_image(x))
	#tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the bottleneck.
  #entropy_bottleneck.visualize()

  #hooks = [
  #    tf.train.StopAtStepHook(last_step=args.last_step),
  #    tf.train.NanTensorHook(train_loss),
  #]
	model = encoder((640,640,3), 6)
	
	#Need to define two custom losses; one will be for distortion (e.g., MS-SSMM), the other for the center loss
	model.compile(optimizer='adam', loss=[msssmm_loss, center_loss], metrics=['accuracy'])
	
	#present layer information
	model.summary()

	#fit model
	model.fit_generator(train_generator, steps_per_epoch = 103/32, epochs = 2)
	
	ep = 0
	epSub = 0
	nm = 100 #Number of Steps Between Each Log Print
	with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=args.checkpoint_dir,
			save_checkpoint_secs=60, save_summaries_secs=60) as sess:
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
	print('TRAIN COMPLETED')



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
	  "--train_glob", default="few_images",
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
		#decompress()


