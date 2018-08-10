import tensorflow as tf
import numpy as np

'''
- This is a code of models and modules for "MIREX2018: Patterns of Prediction".
'''
# modules
def look_up(input_dim, emb_dim, scope):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		lookup_table = tf.get_variable('lookup',
								 dtype=tf.float32,
								 shape=[input_dim, emb_dim],
								 initializer=tf.random_normal_initializer())
		return lookup_table

def GRU(inp, num_unit, scope):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		gru_cell = tf.nn.rnn_cell.GRUCell(num_units=num_unit,
										  kernel_initializer=tf.random_normal_initializer())
		enc_length = length(inp)
		gru_out, gru_states = tf.nn.dynamic_rnn(gru_cell, inp,
												sequence_length=enc_length,
												dtype=tf.float32)
		return gru_out, gru_states

def biGRU(inp, num_unit, scope):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		gru_cell = tf.nn.rnn_cell.GRUCell(num_units=num_unit,
										  kernel_initializer=tf.random_normal_initializer())
		enc_length = length(inp)
		gru_out, gru_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, inp,
													  		  sequence_length=enc_length,
															  dtype=tf.float32)
		return gru_out, gru_states

def conv1d(inp, filters, kernel_size, dilation_rate, padding, scope):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		padded_ = dilation_rate * (kernel_size - 1) / 2
		if padding == "causal":
			pad = tf.constant([[0,0],[int(padded_*2),0],[0,0]])
		elif padding == "same":
			pad = tf.constant([[0,0],[int(padded_),int(padded_)],[0,0]])
		elif padding == "valid":
			pad = tf.constant([[0,0],[0,0],[0,0]])
		padded_input = tf.pad(inp, pad, "CONSTANT")
		conv_out = tf.layers.conv1d(padded_input,
									filters=filters,
									kernel_size=kernel_size,
									strides=1,
									dilation_rate=dilation_rate,
									kernel_initializer=tf.random_normal_initializer())
		return conv_out

def length(seq): # for calculating sample lengths in one mini-batch
	# if each sample contain value other than 0, returns 1
    filled = tf.sign(tf.reduce_max(tf.abs(seq), reduction_indices=2))
    # calculate number of samples of 1 -> this indicates a length of single batch
    length = tf.reduce_sum(filled, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def attention(K, Q, V, d, softmax, scope):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		kq_norm = tf.matmul(K, tf.transpose(Q, perm=[0,2,1])) / np.sqrt(d)
		if softmax is True:
			A = tf.nn.softmax(kq_norm, dim=1, name="softmax")
		else: A = tf.nn.sigmoid(kq_norm, name="sigmoid")
		R = tf.matmul(tf.transpose(A, perm=[0,2,1]), V)
		return R, A


# MODELS
'''
Models candidates:
- 1. DAC-like model(1) (seperate models for dur and mnn)
	- Deep artificial composer: a creative neural network model for automated melody generation(Colombo et al., 2017)
- 2. DAC-like model(2) (seperate models for dur and mnn)
	- Deep artificial composer: a creative neural network model for automated melody generation(Colombo et al., 2017)
- 3. LookbackRNN-like model
	- https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn
'''

# DAC-like model(1)
def mono_1(inp, inp_dim, num_unit, emb_dim, phase, scope):
		# for duration
		def dur_model():
			with tf.variable_scope("dur_model", reuse=tf.AUTO_REUSE):
				gru_out0 = GRU(inp, num_unit, scope="gru_0")
				gru_out1 = GRU(gru_out0, num_unit, scope="gru_1")
				final_out = tf.nn.softmax(gru_out1, name='softmax')
			return final_out
		# for pitch
		def mnn_model():
			with tf.variable_scope("mnn_model", reuse=tf.AUTO_REUSE):
				gru_out2 = GRU(inp, num_unit, scope="gru_2")
				gru_out3 = GRU(gru_out2, num_unit, scope="gru_3")
				final_out = tf.nn.softmax(gru_out3, name='softmax')
			return final_out








# def monophonic_1(inp, inp_dim, num_unit, emb_dim, phase, scope):
# 	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# 		# prime encoder
# 		lookup_table0 = look_up(inp_dim, emb_dim, "prime_lookup")
# 		emb_out0 = tf.nn.embedding_lookup(lookup_table0, inp0, name="emb_0")
# 		conv_out0 = conv1d(emb_out0, num_unit*3, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_0")
# 		bn_out0 = tf.layers.batch_normalization(conv_out0, training=phase, name="bn_0")
# 		relu_out0 = tf.nn.relu(bn_out0, name="relu_0")
# 		K, Q, V = tf.split(relu_out0, 3, axis=-1)
# 		R, A = attention(K, Q, V, d=num_unit, softmax=False, scale=True, scope="attention")
# 		R_ = np.concatenate([R, Q], axis=-1)
# 		gru_out0, gru_states0 = biGRU(R_, num_unit, "biGRU")
# 		bigru_out = tf.concat(gru_out0, axis=-1, name="bigru_out")
# 		conv_out1 = conv1d(bigru_out, inp_dim, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_1")
# 		final_out = tf.nn.softmax(conv_out1, name='softmax')
# 		return final_out, A
#
# def monophonic_2(inp0, inp1, inp_dim, num_unit, emb_dim, phase, scope):
# 	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# 		# prime encoder
# 		lookup_table0 = look_up(inp_dim, emb_dim, "prime_lookup")
# 		emb_out0 = tf.nn.embedding_lookup(lookup_table0, inp0, name="emb_0")
# 		# emb_out0 = tf.einsum('bnk,kd->bnd', inp0, lookup_table0, name="emb_0")
# 		# gru_out0, gru_states0 = biGRU(emb_out0, num_unit, "biGRU")
# 		# bigru_out = tf.concat(gru_out0, axis=-1, name="bigru_out")
# 		conv_out0 = conv1d(emb_out0, num_unit*2, kernel_size=3, dilation_rate=1, padding="valid", scope="conv_0")
# 		conv_out1 = conv1d(conv_out0, num_unit*2, kernel_size=3, dilation_rate=3, padding="same", scope="conv_1")
# 		conv_out2 = conv1d(conv_out1, num_unit*2, kernel_size=3, dilation_rate=9, padding="same", scope="conv_2")
# 		bn_out0 = tf.layers.batch_normalization(conv_out2, training=phase, name="bn_0")
# 		relu_out0 = tf.nn.relu(bn_out0, name='relu_0')
# 		res_out0 = relu_out0 + conv_out0
# 		K, V = tf.split(res_out0, 2, axis=-1)
# 		# cont encoder
# 		lookup_table1 = look_up(inp_dim, emb_dim, "cont_lookup")
# 		emb_out1 = tf.nn.embedding_lookup(lookup_table1, inp1, name="emb_1")
# 		# emb_out1 = tf.einsum('bnk,kd->bnd', inp1, lookup_table1, name="emb_1")
# 		# gru_out1, gru_states1 = GRU(emb_out1, num_unit, "gru_1")
# 		# conv_out1 = conv1d(bigru_out, num_unit, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_1")
# 		conv_out3 = conv1d(emb_out1, num_unit, kernel_size=3, dilation_rate=1, padding="valid", scope="conv_3")
# 		conv_out4 = conv1d(conv_out3, num_unit, kernel_size=3, dilation_rate=3, padding="causal", scope="conv_4")
# 		conv_out5 = conv1d(conv_out4, num_unit, kernel_size=3, dilation_rate=9, padding="causal", scope="conv_5")
# 		bn_out1 = tf.layers.batch_normalization(conv_out5, training=phase, name="bn_1")
# 		relu_out1 = tf.nn.relu(bn_out1, name='relu_1')
# 		res_out1 = relu_out1 + conv_out3
# 		Q = relu_out1
# 		# attention
# 		R, A = attention(K, Q, V, d=num_unit, softmax=False, scale=True, scope="attention")
# 		R_ = tf.concat([R, Q], axis=-1)
# 		# cont decoder
# 		# gru_out2, gru_states2 = GRU(R_, inp_dim, "gru_2")
# 		conv_out6 = conv1d(R_, num_unit, kernel_size=3, dilation_rate=1, padding="valid", scope="conv_6")
# 		conv_out7 = conv1d(conv_out6, num_unit, kernel_size=3, dilation_rate=3, padding="causal", scope="conv_7")
# 		conv_out8 = conv1d(conv_out7, num_unit, kernel_size=3, dilation_rate=9, padding="causal", scope="conv_8")
# 		bn_out2 = tf.layers.batch_normalization(conv_out8, training=phase, name="bn_2")
# 		relu_out2 = tf.nn.relu(bn_out2, name='relu_2')
# 		res_out3 = relu_out2 + conv_out6
# 		conv_out9 = conv1d(res_out3, inp_dim, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_9")
# 		bn_out3 = tf.layers.batch_normalization(conv_out9, training=phase, name="bn_3")
# 		final_out = tf.nn.softmax(bn_out3, name='softmax')
# 		return final_out, A
#
# def monophonic_3(inp0, inp1, inp_dim, num_unit, emb_dim, phase, scope):
# 	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# 		with tf.variable_scope("input_encoder", reuse=tf.AUTO_REUSE):
# 			# prime encoder
# 			lookup_table0 = look_up(inp_dim, emb_dim, "prime_lookup")
# 			emb_out0 = tf.nn.embedding_lookup(lookup_table0, inp0, name="emb_0")
# 			# emb_out0 = tf.einsum('bnk,kd->bnd', inp0, lookup_table0, name="emb_0")
# 			conv_out0 = conv1d(emb_out0, num_unit*2, kernel_size=3, dilation_rate=1, padding="same", scope="conv_0")
# 			conv_out1 = conv1d(conv_out0, num_unit*2, kernel_size=3, dilation_rate=3, padding="same", scope="conv_1")
# 			conv_out2 = conv1d(conv_out1, num_unit*2, kernel_size=3, dilation_rate=9, padding="same", scope="conv_2")
# 			bn_out0 = tf.layers.batch_normalization(conv_out2, training=phase, name="bn_0")
# 			relu_out0 = tf.nn.relu(bn_out0, name='relu_0')
# 			res_out0 = relu_out0 + conv_out0
# 			K, V = tf.split(res_out0, 2, axis=-1)
# 		with tf.variable_scope("output_encoder", reuse=tf.AUTO_REUSE):
# 			# cont encoder
# 			lookup_table1 = look_up(inp_dim, emb_dim, "cont_lookup")
# 			emb_out1 = tf.nn.embedding_lookup(lookup_table1, inp1, name="emb_1")
# 			# emb_out1 = tf.einsum('bnk,kd->bnd', inp1, lookup_table1, name="emb_1")
# 			conv_out3 = conv1d(emb_out1, num_unit, kernel_size=3, dilation_rate=1, padding="causal", scope="conv_3")
# 			conv_out4 = conv1d(conv_out3, num_unit, kernel_size=3, dilation_rate=3, padding="causal", scope="conv_4")
# 			conv_out5 = conv1d(conv_out4, num_unit, kernel_size=3, dilation_rate=9, padding="causal", scope="conv_5")
# 			bn_out1 = tf.layers.batch_normalization(conv_out5, training=phase, name="bn_1")
# 			relu_out1 = tf.nn.relu(bn_out1, name='relu_1')
# 			res_out1 = relu_out1 + conv_out3
# 			Q = res_out1
# 		with tf.variable_scope("output_decoder", reuse=tf.AUTO_REUSE):
# 			# attention
# 			R, A = attention(K, Q, V, d=num_unit, softmax=False, scale=True, scope="attention")
# 			R_ = tf.concat([R, Q], axis=-1)
# 			# cont decoder
# 			# gru_out2, gru_states2 = GRU(R_, inp_dim, "gru_2")
# 			conv_out6 = conv1d(R_, num_unit, kernel_size=3, dilation_rate=1, padding="causal", scope="conv_6")
# 			conv_out7 = conv1d(conv_out6, num_unit, kernel_size=3, dilation_rate=3, padding="causal", scope="conv_7")
# 			conv_out8 = conv1d(conv_out7, num_unit, kernel_size=3, dilation_rate=9, padding="causal", scope="conv_8")
# 			bn_out2 = tf.layers.batch_normalization(conv_out8, training=phase, name="bn_2")
# 			relu_out2 = tf.nn.relu(bn_out2, name='relu_2')
# 			res_out3 = relu_out2 + conv_out6
# 			conv_out9 = conv1d(res_out3, inp_dim, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_9")
# 			bn_out3 = tf.layers.batch_normalization(conv_out9, training=phase, name="bn_3")
# 			final_out = tf.nn.softmax(bn_out3, name='softmax')
# 		return final_out, A
#
# def monophonic_4(inp0, inp1, inp_dim, num_unit, emb_dim, phase, scope):
# 	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# 		with tf.variable_scope("input_encoder", reuse=tf.AUTO_REUSE):
# 			# prime encoder
# 			lookup_table0 = look_up(inp_dim, emb_dim, "prime_lookup")
# 			# emb_out0 = tf.nn.embedding_lookup(lookup_table0, inp0, name="emb_0")
# 			emb_out0 = tf.einsum('bnk,kd->bnd', inp0, lookup_table0, name="emb_0")
# 			conv_out0 = conv1d(emb_out0, num_unit*3, kernel_size=3, dilation_rate=1, padding="same", scope="conv_0")
# 			conv_out1 = conv1d(conv_out0, num_unit*3, kernel_size=3, dilation_rate=3, padding="same", scope="conv_1")
# 			conv_out2 = conv1d(conv_out1, num_unit*3, kernel_size=3, dilation_rate=9, padding="same", scope="conv_2")
# 			bn_out0 = tf.layers.batch_normalization(conv_out2, training=phase, name="bn_0")
# 			relu_out0 = tf.nn.relu(bn_out0, name='relu_0')
# 			res_out0 = relu_out0 + conv_out0
# 			K, Q, V = tf.split(res_out0, 3, axis=-1)
# 			# self attention
# 			R0, A0 = attention(K, Q, V, d=num_unit, softmax=False, scale=True, scope="attention")
# 			R_0 = tf.add(R0, Q, name="self_att0")
# 			R_1 = tf.identity(R_0, name="self_att1")
# 		with tf.variable_scope("output_encoder", reuse=tf.AUTO_REUSE):
# 			# cont encoder
# 			lookup_table1 = look_up(inp_dim, emb_dim, "cont_lookup")
# 			# emb_out1 = tf.nn.embedding_lookup(lookup_table1, inp1, name="emb_1")
# 			emb_out1 = tf.einsum('bnk,kd->bnd', inp1, lookup_table1, name="emb_1")
# 			conv_out3 = conv1d(emb_out1, num_unit, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_3")
# 			# conv_out4 = conv1d(conv_out3, num_unit, kernel_size=3, dilation_rate=3, padding="causal", scope="conv_4")
# 			# conv_out5 = conv1d(conv_out4, num_unit, kernel_size=3, dilation_rate=9, padding="causal", scope="conv_5")
# 			bn_out1 = tf.layers.batch_normalization(conv_out3, training=phase, name="bn_1")
# 			relu_out1 = tf.nn.relu(bn_out1, name='relu_1')
# 			res_out1 = relu_out1 + conv_out3
# 			Q_ = res_out1
# 			R1, A1 = attention(R_0, Q_, R_1, d=num_unit, softmax=False, scale=True, scope="attention")
# 			R_2 = tf.concat([R1, Q_], axis=-1)
# 		with tf.variable_scope("output_decoder", reuse=tf.AUTO_REUSE):
# 			# cont decoder
# 			conv_out6 = conv1d(R_2, num_unit, kernel_size=1, dilation_rate=1, padding="causal", scope="conv_6")
# 			# conv_out7 = conv1d(conv_out6, num_unit, kernel_size=3, dilation_rate=3, padding="causal", scope="conv_7")
# 			# conv_out8 = conv1d(conv_out7, num_unit, kernel_size=3, dilation_rate=9, padding="causal", scope="conv_8")
# 			bn_out2 = tf.layers.batch_normalization(conv_out6, training=phase, name="bn_2")
# 			relu_out2 = tf.nn.relu(bn_out2, name='relu_2')
# 			res_out3 = relu_out2 + conv_out6
# 			conv_out9 = conv1d(res_out3, inp_dim, kernel_size=1, dilation_rate=1, padding="valid", scope="conv_9")
# 			bn_out3 = tf.layers.batch_normalization(conv_out9, training=phase, name="bn_3")
# 			final_out = tf.nn.softmax(bn_out3, name='softmax')
# 		return final_out, A1
