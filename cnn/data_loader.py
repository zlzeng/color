import numpy as np

import tensorflow as tf

import cv2

from scipy.misc import imread, imsave, imresize

import os
import sys
import glob
import time
import random

train_file = '../dataset/train_c.txt'
test_file = '../dataset/test_c.txt'

def data_loader(batch_size=1, file=train_file, resize=False):
	"""
	Read pair of training set
	Use fixed input size: [512x1024x3], gt size: [40x1024x3]
	"""
	paths = open(file, 'r').read().splitlines()

	random.shuffle(paths)

	image_paths = [p.split('\t')[0] for p in paths]
	label_paths = [p.split('\t')[1] for p in paths]

	# create batch input
	# convert to tensor list
	img_list  = tf.convert_to_tensor(image_paths, dtype=tf.string)
	lab_list = tf.convert_to_tensor(label_paths, dtype=tf.string)

	# create data queue
	data_queue = tf.train.slice_input_producer([img_list, lab_list], 
		shuffle=False, capacity=batch_size*128)

	# decode image
	image = tf.image.decode_png(tf.read_file(data_queue[0]), channels=3)
	label = tf.image.decode_png(tf.read_file(data_queue[1]), channels=3)

	# resize to define image shape
	if not resize:
		image = tf.reshape(image, [512, 1024, 3])
		label = tf.reshape(label, [40, 1024, 3])
	else:
		image = tf.image.resize_images(image, (128, 256))
		label = tf.image.resize_images(label, (10, 256))

	# convert to float data type
	image = tf.cast(image, dtype=tf.float32)	
	label  = tf.cast(label, dtype=tf.float32)

	# data pre-processing, normalize
	image = tf.divide(image, tf.constant(255.0))
	label = tf.divide(label, tf.constant(255.0))

	# one-hot label, convert to one-hot label during loss computation
	# label = 

	# create batch data
	images, labels = tf.train.shuffle_batch([image, label],
		batch_size=batch_size, num_threads=1, 
		capacity=batch_size*128, min_after_dequeue=batch_size*32) 

	# number of batches
	num_batch = len(image_paths) // batch_size	

	return {'images':images, 'labels':labels, 'num_batch':num_batch}

# Unit test
# if __name__ == '__main__':
# 	data_dict = data_loader()

# 	images = data_dict['images']
# 	labels = data_dict['labels']
# 	print images.shape, labels.shape

# 	sess = tf.Session()
# 	sess.run(tf.group(tf.global_variables_initializer(),
# 					tf.local_variables_initializer()))

# 	# coordinator for queue runner
# 	coord = tf.train.Coordinator()

# 	# start queue 
# 	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 	batch_im, batch_gt = sess.run([images, labels])
# 	from matplotlib import pyplot as plt
# 	plt.subplot(121)
# 	plt.imshow(batch_im[0,...])
# 	plt.subplot(122)
# 	plt.imshow(batch_gt[0,...])
# 	plt.show()
