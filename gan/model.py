from data_loader import *
from net import *


seed = 8964
tf.set_random_seed(seed)

class Model(object):
	"""docstring for Baseline"""
	def __init__(self):
		self.logdir='./LOG'	

	def build_train_graph(self):

		# get inputs
		data_dict = data_loader(batch_size=1)
		images = data_dict['images']
		labels = data_dict['labels']

		# initial generator 
		logits_gen, _ = gen(images)

		# fake_pair = tf.concat([images, tf.nn.sigmoid(logits_gen)], axis=1) # 1x(512+40)x1024x3
		# real_pair = tf.concat([images, labels], axis=1)

		# initial discriminator
		logits_fake, _ = dis(tf.nn.sigmoid(logits_gen))
		logits_true, _ = dis(labels)

		# compute loss 
		self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_gen, labels=labels, name='gen_loss'))
		# self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones([1,1]), name='gen_loss'))
		
		t_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_true, labels=tf.ones([1,1]), name='dis_true')) 
		f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros([1,1]), name='dis_fake'))
		self.dis_loss = (t_loss + f_loss) / 2.0

		# compute loss 2
		# self.gen_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(logits_fake)+1e-6))
		# self.dis_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(logits_true)+1e-6)-tf.log(1-tf.nn.sigmoid(logits_fake)+1e-6))

		# create train op
		g_vars = [v for v in tf.trainable_variables() if v.name.startswith('gen')]
		d_vars = [v for v in tf.trainable_variables() if v.name.startswith('dis')]
		# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		self.g_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.gen_loss, var_list=g_vars, colocate_gradients_with_ops=True) # gradient ops assign to same device as forward ops
		self.d_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.dis_loss, var_list=d_vars, colocate_gradients_with_ops=True) 

		# collect summaries
		tf.summary.image('input', images)
		tf.summary.image('label', labels)
		tf.summary.image('predict', tf.nn.sigmoid(logits_gen)) 	
		tf.summary.scalar('gen loss', self.gen_loss)		
		tf.summary.scalar('dis loss', self.dis_loss)		

		return data_dict['num_batch']

	def train(self, max_step=40000):
		# build train graph
		num_batch = self.build_train_graph()

		max_ep = max_step // num_batch

		# create session and start train session
		config = tf.ConfigProto(allow_soft_placement=True) 
		config.gpu_options.allow_growth=True # prevent the program occupies all GPU memory
		with tf.Session(config=config) as sess:
			# init all variables in graph
			sess.run(tf.group(tf.global_variables_initializer(),
							tf.local_variables_initializer()))

			# saver 
			saver = tf.train.Saver([v for v in tf.trainable_variables() if 'gen' in v.name], max_to_keep=20) # normal model and extra model total 20

			# filewriter for log info
			log_dir = self.logdir+'/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
			writer = tf.summary.FileWriter(log_dir)
			merged = tf.summary.merge_all()

			# coordinator for queue runner
			coord = tf.train.Coordinator()

			# start queue 
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			print "Start Training!"
			total_times = 0			

			for ep in xrange(max_ep): # epoch loop
				for n in xrange(num_batch): # batch loop
					tic = time.time()
					[d_loss_value, update_value] = sess.run([self.dis_loss, self.d_optim])	
					[g_loss_value, update_value, summaries] = sess.run([self.gen_loss, self.g_optim, merged])	
					duration = time.time()-tic

					total_times += duration

					step = int(ep*num_batch + n)
					# write log 
					print 'step {}: loss = {:.3}; {:.2} data/sec, excuted {} minutes'.format(step,
						(g_loss_value+d_loss_value), 1.0/duration, int(total_times/60))
					if ep % 5 == 0:
						writer.add_summary(summaries, global_step=step)
				# save model parameters after 20 epoch training
				if ep % 20 == 0:
					saver.save(sess, self.logdir+'/model', global_step=ep)
			saver.save(sess, self.logdir+'/model', global_step=max_ep)

			# close session
			coord.request_stop()
			coord.join(threads)			
			sess.close()	

	def setup_inference(self, shape=[1,512,1024,3]):
		"""
		Use both inference and offline evaluation
		"""
		self.x = tf.placeholder(shape=shape, dtype=tf.float32)
		logits, _ = gen(self.x)
		self.infer = tf.nn.sigmoid(logits)

	def inference(self, image, sess):
		im = np.reshape(image, (1, 512, 1024, 3))
		
		pred = sess.run([self.infer], feed_dict={self.x:im})
		return np.squeeze(pred)