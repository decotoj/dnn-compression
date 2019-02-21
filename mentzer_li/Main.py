import argparse
import glob
import time
import numpy as np
import tensorflow as tf
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import keras.backend as K

def residual_block(X, f, filter_size, stage):
	conv_name_base = 'res' + str(stage)
	bn_name_base = 'bn' + str(stage)
	X_shortcut = X
	#1/3 residual blocks - no Relu at end?
	X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + 'a', kernel_initializer = glorot_uniform(seed=None))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + 'a')(X)
	X = Activation('relu')(X)
	X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + 'b', kernel_initializer = glorot_uniform(seed=None))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + 'b')(X)
	
	#2/3 residual blocks - no Relu at end?
	X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + 'c', kernel_initializer = glorot_uniform(seed=None))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + 'c')(X)
	X = Activation('relu')(X)
	X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + 'd', kernel_initializer = glorot_uniform(seed=None))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + 'd')(X)

	#3/3 residual blocks - no Relu at end?
	X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + 'e', kernel_initializer = glorot_uniform(seed=None))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + 'e')(X)
	X = Activation('relu')(X)
	X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + 'f', kernel_initializer = glorot_uniform(seed=None))(X)
	
	#skip connection - no Relu I think
	X = Add()([X,X_shortcut])
	return X


def encoder(x, code_numbers):
	
	#ENCODING
	X_input = Input(shape = x)
	
	# Stage 1
	X = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', kernel_initializer = glorot_uniform(seed=None))(X_input)
	X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
	X = Activation('relu')(X)
	
	X = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same', kernel_initializer = glorot_uniform(seed=None))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
	X = Activation('relu')(X)

	# 1/5 Residual block 
	X = residual_block(X, 3, 128, stage=1)

	# 2/5 Residual block 
	X = residual_block(X, 3, 128, stage=2)

	# 3/5 Residual block 
	X = residual_block(X, 3, 128, stage=3)

	# 4/5 Residual block 
	X = residual_block(X, 3, 128, stage=4)

	# 5/5 Residual block 
	X = residual_block(X, 3, 128, stage=5)

	X = BatchNormalization(axis = 3, name = 'bn_conv6')(X)
	X = Conv2D(code_numbers, (5,5), strides = (2,2), padding = 'same', kernel_initializer = glorot_uniform(seed=None))(X)

	#this gets fed into our importance map
	importance_map = importance_map(X)
	X_encode = mask_with_importance(X,importance_map)

	#now we can quantize the masked encoding

	#QUANTIZATION - centers isn't defined anywhere yet.  Need some kind of Input to generate this.
	X_quantized = quantizer(X_encode,centers)
		
	#once quantized we can decode X_quantized

	#return Encoder model
	model = Model(inputs = X_input, outputs = X, name='encoder')
	return model

#quantization
def quantizer(X_encode, centers):
	#probably need to adjust shape of X_encode in some capacity
	dist = tf.square(tf.abs(X_encode - centers))
	phi_hard = tf.nn.softmax(-1e7 * dist, dim=-1)
	symbols_hard = tf.argmax(phi_hard, axis=-1)
	phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)
	matmul_innerproduct = phi_hard * centers  # (B, C, m, L)
    return tf.reduce_sum(matmul_innerproduct, axis=3)  # (B, C, m)

#adapted from github, not sure how to modify
def importance_map(bottleneck):
	assert bottleneck.shape.ndims == 4, bottleneck.shape
	C = int(bottleneck.shape[1]) - 1  # -1 because first channel is heatmap
	heatmap_channel = Input(bottleneck[:, 0, :, :]) # I assume you need?
	heatmap_channel = bottleneck[:, 0, :, :]  # NHW
		#heatmap2D = tf.nn.sigmoid(heatmap_channel) * C  # NHW

	heatmap2D = Activation('sigmoid')(heatmap_channel)*C
	c = tf.range(C, dtype=tf.float32)  # C
	# reshape heatmap2D for broadcasting
	heatmap = tf.expand_dims(heatmap2D, 1)  # N1HW
	# reshape c for broadcasting
	c = tf.reshape(c, (C, 1, 1))  # C11
	# construct heatmap3D
	# if heatmap[x, y] == C, then heatmap[x, y, c] == 1 \forall c \in {0, ..., C-1}
	heatmap3D = tf.maximum(tf.minimum(heatmap - c, 1), 0, name='heatmap3D')  # NCHW
	return heatmap3D

#adapted from github, not sure how to modify
def mask_with_importance(bottleneck, heatmap3D):
	bottleneck_without_heatmap = bottleneck[:, 1:, ...]
	return heatmap3D * bottleneck_without_heatmap

def decode(x):

	X_encode = Input(shape = x)
	#DECODING
	X = Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = glorot_uniform(seed=None))(X_encode)
	X = BatchNormalization(axis = 3, name = 'bn_conv7')(X)
	X = Activation('relu')(X)

	# 1/5 Residual block 
	X = residual_block(X, 3, 128, stage=8)

	# 2/5 Residual block 
	X = residual_block(X, 3, 128, stage=9)

	# 3/5 Residual block 
	X = residual_block(X, 3, 128, stage=10)

	# 4/5 Residual block 
	X = residual_block(X, 3, 128, stage=11)

	# 5/5 Residual block 
	X = residual_block(X, 3, 128, stage=12)

	X = BatchNormalization(axis = 3, name = 'bn_conv13')(X)
	X = Conv2D(64, (5,5), strides = (2,2), padding = 'same', kernel_initializer = glorot_uniform(seed=None))(X)
	X = Activation('relu')(X)

	X = Conv2D(3, (5, 5), strides = (2, 2), padding = 'same', kernel_initializer = glorot_uniform(seed=None))(X)
	
	model = Model(inputs = X_encode, outputs = X, name='decoder')
	return model









