from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


def cnn(inputs):
	with tf.variable_scope('feature_net') as sc:						  # design the nn architecture for the depth network
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],	   #define a conv2d operator with fixed params shown below
							normalizer_fn=None,
							weights_regularizer=slim.l2_regularizer(0.05), # using l2 regularizer with 0.05 weight
							activation_fn=tf.nn.relu,
							outputs_collections=end_points_collection):

			#for slim.conv2d the default padding mode = 'same'
			cnv1  = slim.conv2d(inputs, 32,  [7, 7], stride=2, scope='cnv1') #1*256*512*32
			cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
			cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')	 #1*128*256*64
			cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
			cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')	 #1*64*128*128
			cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
			cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')	 #1*32*64*256
			cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
			cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
			cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')	# 1*16*32*256

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
