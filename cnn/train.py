from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warning messages from tf package

def main(_):
	seed = 8964
	tf.set_random_seed(seed)
	np.random.seed(seed)	

	model = Model()
	model.train()


if __name__ == '__main__':
	tf.app.run()
