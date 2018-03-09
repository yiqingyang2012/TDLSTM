import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops import init_ops

class TDLSTMmodel(object):
	"""docstring for model"""
	def __init__(self, embeding_matrix, parameters):
		self.params = parameters
		self.embeding_matrix = embeding_matrix
		self.global_step = tf.Variable(0, trainable=False)
		self._add_placeholders()
		embedding = self.get_embedding(self.embeding_matrix)
		with tf.variable_scope("embeding_lookup"):
			emb_right = tf.nn.embedding_lookup(embedding, self._rightText)
			emb_left = tf.nn.embedding_lookup(embedding, self._leftText)
			emb_aspects = tf.nn.embedding_lookup(embedding, self._aspects)
		
		with tf.variable_scope("concat_asepct"):
			float_aspect_len = tf.cast(self._actual_aspects_lens, tf.float64)
			transspose_aspect_len = tf.expand_dims(tf.tile(tf.expand_dims(float_aspect_len, 1), [1, self.params.embedingDim]), 1)
			
			sumAspect = tf.reduce_sum(emb_aspects, 1, keep_dims=True) / transspose_aspect_len
			print("sumAspect shape")
			print( sumAspect.get_shape().as_list())
			words_represention = tf.tile(sumAspect, [1, self.params.max_article_len, 1])
			print("words_represention shape")
			print( words_represention.get_shape().as_list())
			
		with tf.variable_scope("concat_aspect_with_every_word"):
			left_sequence = tf.concat([emb_left, words_represention], 2)
			right_sequence = tf.concat([emb_right, words_represention], 2)
			tf.summary.histogram("left_concat", left_sequence)
			tf.summary.histogram("right_concat", right_sequence)
			
		with tf.variable_scope("LSTM"):
			fw_cell = LSTMCell(num_units = self.params.word_hidden_dim,
											initializer = tf.random_uniform_initializer(-0.003, 0.003, seed=123), state_is_tuple = True)
			bw_cell = LSTMCell(num_units = self.params.word_hidden_dim,
											initializer = tf.random_uniform_initializer(-0.003, 0.003, seed=123), state_is_tuple = True)
			fw_init_state = fw_cell.zero_state(self.params.batch_size, dtype=tf.float64)
			bw_init_state = fw_cell.zero_state(self.params.batch_size, dtype=tf.float64)
			
			print( [state.get_shape().as_list() for state in fw_init_state])
			print( [state.get_shape().as_list() for state in bw_init_state])
			#length64 = tf.cast(self.lengths, tf.int64)
			_, fw_last_state = tf.nn.dynamic_rnn(
					fw_cell,
					left_sequence,
					initial_state = fw_init_state,
					dtype=tf.float64,
					sequence_length=self._actual_left_text_lens,
					scope="fw"
				)
			_, bw_last_state = tf.nn.dynamic_rnn(
					bw_cell,
					right_sequence,
					initial_state = bw_init_state,
					dtype=tf.float64,
					sequence_length=self._actual_right_text_lens,
					scope="bw"
				)
		print("yiqingyang")
		#print( bw_last_state[1].get_shape().as_list())
		#print( fw_last_state[1].get_shape().as_list())
		with tf.variable_scope("concat_lstm_hidden_value"):
			final_concat_state = tf.concat([fw_last_state[1], bw_last_state[1]], 1)
			
			tf.summary.histogram("concat_the_final_hid_state", final_concat_state)
		
		with tf.variable_scope("project_loss"):
			with tf.name_scope('weights'):
				w1 = tf.get_variable(
						'W1',
						[self.params.word_hidden_dim*2 , 3],
						initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64), dtype=tf.float64)
			with tf.name_scope('biases'):
				b1 = tf.get_variable('b_project', initializer=tf.constant(0.1, shape=[3], dtype=tf.float64))
		
			scores = tf.matmul(final_concat_state, w1) + b1
			losse_vector = tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = self.polarity)
			self.loss = tf.reduce_mean(losse_vector)
			tf.summary.scalar("loss", self.loss)
			self.predictions = tf.argmax(scores, 1, name="predictions")
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.polarity, 1))
		
		#with tf.variable_scope("grad_param"):
		train_params = tf.trainable_variables()
		opt = tf.train.AdadeltaOptimizer( learning_rate = self.params.learning_rate, epsilon = 1e-6)
		#grads_vars include grad and value, and can iterate
		grads_vars = opt.compute_gradients(self.loss)
		with tf.variable_scope("grad_param"):
			grads = [i[0] for i in grads_vars]
		'''
		grad_summaries = []
		for g, v in grads_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)
		'''

		with tf.variable_scope("clipped_grad_and_apply"):
			#norm and clip gredients, max_gradient is a manul setted threshold para
			#clipped_gradients = [(tf.clip_by_value(g, -self.params.clip, self.params.clip), v)
			#					 for g, v in grads_vars]
			clipped_gradients, norm = \
						tf.clip_by_global_norm(grads, self.params.max_gradient)
			'''
			grad_summaries = []
			for g, v in zip(clipped_gradients, train_params):
				if g is not None:
					grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.summary.merge(grad_summaries)
			'''
			#update grediets
			self.train_op = opt.apply_gradients(zip(clipped_gradients, train_params), global_step=self.global_step)
		
		self.merged = tf.summary.merge_all()

		self.saver = tf.train.Saver(tf.global_variables())
		
		'''
		# optimizer of the model
		self.opt = tf.train.AdamOptimizer(self.params.learning_rate)
		# apply grad clip to avoid gradient explosion
		grads_vars = self.opt.compute_gradients(self.loss)
		capped_grads_vars = [(tf.clip_by_value(g, -self.params.clip, self.params.clip), v)
							 for g, v in grads_vars]  # gradient capping
		self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
		self.saver = tf.train.Saver(tf.global_variables())
		'''
	def _add_placeholders(self):
	  	hps = self.params
	  	with tf.name_scope('input'):
		  	self._rightText = tf.placeholder(tf.int64, [None, hps.max_article_len], name='rightText')
		  	self._leftText = tf.placeholder(tf.int64,
		                                    [None, hps.max_article_len],
		                                    name='leftText')
		  	self._aspects = tf.placeholder(tf.int64,
		                                   [None, hps.max_target_len],
		                                   name='targets')
		  	self._actual_right_text_lens = tf.placeholder(tf.int64, [None],
		                                        name='actual_right_lens')
		  	self._actual_left_text_lens = tf.placeholder(tf.int64, [None],
		                                        name='actual_left_lens')
		  	self._actual_aspects_lens = tf.placeholder(tf.int64, [None],
		                                         name='aspects_lens')
		  	self.polarity = tf.placeholder(tf.float64,
		                                        [None, 3],
		                                        name='batch_polarity')

	def get_embedding(self, embedingVec):
		with tf.variable_scope("Embedding"), tf.device('/cpu:0'):
			embedding = tf.get_variable("word_emb",
										initializer=embedingVec,
										dtype=np.float64,
										trainable = False)

		return embedding
	
	def runstep(self, sess, batch, istrain):
		feed_dict = {
			self._leftText : batch['left'],
            self._rightText : batch['right'],
            self._aspects : batch['aspects'],
            self._actual_right_text_lens : batch['rightlen'],
            self._actual_left_text_lens : batch['leftlen'],
            self._actual_aspects_lens : batch['aspectlen'],
            self.polarity : batch['polarity']
			}
		if istrain:
			return sess.run([self.loss, self.train_op, self.merged], feed_dict=feed_dict)
			
		else:
			assert batch['left'].shape == batch['right'].shape, print("error left{0}, right:{1}".format(batch['left'].shape, batch['right'].shape))
			return sess.run([self.predictions, self.correct_predictions], feed_dict=feed_dict)

