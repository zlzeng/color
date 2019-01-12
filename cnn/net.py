from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils

def atrous_spatial_pyramid_pooling(inputs, output_stride=16, depth=256):
 	"""Atrous Spatial Pyramid Pooling.

  	Args:
		inputs: A tensor of size [batch, height, width, channels].
		output_stride: Determines the rates for atrous convolution.
	  		the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
		batch_norm_decay: The moving average decay when estimating layer activation
	  		statistics in batch normalization.
		is_training: A boolean denoting whether the input is for training.
		depth: The depth of the ResNet unit output.

  	Returns:
		The atrous spatial pyramid pooling output.
	"""
 	with tf.variable_scope("aspp"):
		if output_stride not in [8, 16]:
	  		raise ValueError('output_stride must be either 8 or 16.')

		atrous_rates = [6, 12, 18]
		if output_stride == 8:
			atrous_rates = [2*rate for rate in atrous_rates]

		inputs_size = tf.shape(inputs)[1:3]
		# (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
		# the rates are doubled when output stride = 8.
		conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
		conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
		conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
		conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

		# (b) the image-level features
		with tf.variable_scope("image_level_features"):
			# global average pooling, chenage keepdims => keep_dims for tensorflow(<1.6)
			image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
			# 1x1 convolution with 256 filters( and batch normalization)
			image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
			# bilinearly upsample features
			image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

		net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
		net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

		return net


def cnn(inputs, is_training=None):
	with tf.variable_scope('feature_net') as sc:						  # design the nn architecture for the depth network
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],	   #define a conv2d operator with fixed params shown below
							normalizer_fn=None,
							weights_regularizer=slim.l2_regularizer(0.05), # using l2 regularizer with 0.05 weight
							activation_fn=tf.nn.relu,
							outputs_collections=end_points_collection):

			#for slim.conv2d the default padding mode = 'same'
			cnv1  = slim.conv2d(inputs, 32,  [7, 7], stride=2, scope='cnv1') 
			cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
			cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')	 
			cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
			cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')	 
			cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
			cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')	 
			cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
			cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5') 
			cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')	

			# aspp_pool = atrous_spatial_pyramid_pooling(cnv4b)

			with tf.variable_scope('fc'):
				cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=1, scope='cnv6')
				cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
				legend_pred = slim.conv2d(cnv6b, 3, [1, 1], stride=1, scope='pred',
					normalizer_fn=None, activation_fn=None)

				legend_pred_resize = tf.image.resize_images(legend_pred, [40, 1024])

			end_points = utils.convert_collection_to_dict(end_points_collection)
			return legend_pred_resize, end_points

# unit test of the tensor shape at each layer
# if __name__ == '__main__':
# 	logits, end_points = cnn(tf.placeholder(shape=[1,512,1024,3], dtype=tf.float32))

# 	keys = end_points.keys()
# 	for k in keys:
# 		print 'tensor name = {}, shape = {}'.format(k, end_points[k].shape)
# 	print 'predict logits shape = ', logits.shape
