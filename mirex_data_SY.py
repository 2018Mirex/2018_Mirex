import numpy as np
import os
import re
import sys
from glob import glob
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

'''
- This is a code for parsing symbolic data for "MIREX2018: Patterns of Prediction".
- Only 'monophonic' version works.
'''

def parse_info():
	datapath = '/home/rsy/Documents/MIREX/dataset/PPDD_original/ \
				PPDD-Jul2018_sym_mono_small/PPDD-Jul2018_sym_mono_small'
	datapath2 = '/home/rsy/Documents/MIREX/dataset/PPDD_mono_small'
	foil_path = os.path.join(datapath, 'cont_foil_csv')
	true_path = os.path.join(datapath, 'cont_true_csv')
	prime_path = os.path.join(datapath, 'prime_csv')
	all_path = [prime_path, true_path, foil_path]
	datalen_list = dict()
	mnn_list, pos_list, dur_list, chn_list = dict(), dict(), dict(), dict()
	file2ind, ind2file = dict(), dict()
	print('---PARSING CSV FILES---')
	# parse prime, true and foil files
	for path in all_path: # each type of path [prime, true, foil]
		typename = path.split('/')[-1][:-4]
		filelist = sorted(glob(os.path.join(path, "*.csv"))) # brings all file lists
		datalen_list[typename] = list()
		mnn_list[typename] = dict()
		dur_list[typename] = dict()
		pos_list[typename] = dict()
		chn_list[typename] = dict()
		savepath = os.path.join(datapath2, '%s_npy' % typename)
		if not os.path.exists(savepath): # if path does not exist
			os.makedirs(savepath) # create one
		for fi, each_file in enumerate(filelist): # brings each file list
			ind = ind2string(fi, 3)
			filename = each_file.split('/')[-1]
			file2ind[filename] = ind
			ind2file[ind] = filename
			f = open(os.path.join(path, filename), 'r') # open file
			data = f.readlines() # read file
			each_file_data = list()
			datalen = 0
			for d, each_data in enumerate(data): # brings each line in csv data
				splited_ = re.split(',|\n', each_data) # split by ',' and '\n'
				s = [float(ss) for ss in splited_[:-1]] # without "" at the end
				try:
					mnn_list[typename][s[1]] += 1 # MNN list for histogram
					dur_list[typename][np.round(s[3],2)] += 1 # dur list for histogram
					pos_list[typename][np.round(s[0]-int(s[0]),2)] += 1 # dur list for histogram
					chn_list[typename][s[4]] += 1
				except:
					mnn_list[typename][float_s[1]] = 1
					dur_list[typename][np.round(s[3],2)] = 1
					pos_list[typename][np.round(s[0]-int(s[0]),2)] = 1
					chn_list[typename][s[4]] = 1
				each_file_data.append(s)
				datalen += 1
			datalen_list[typename].append(datalen) # datalen list to figure out how long
			np.save(os.path.join(savepath, '%s_mono_raw.npy' % ind), each_file_data)
			f.close()
			print('parsing %ith csv files' % (fi+1), end='\r')
		print('parsed %s csv files' % typename)
	# save npy files
	np.save(os.path.join(datapath2, 'datalen_list.npy'), datalen_list)
	np.save(os.path.join(datapath2, 'mnn_list.npy'), mnn_list)
	np.save(os.path.join(datapath2, 'dur_list.npy'), dur_list)
	np.save(os.path.join(datapath2, 'pos_list.npy'), pos_list)
	np.save(os.path.join(datapath2, 'chn_list.npy'), chn_list)
	np.save(os.path.join(datapath2, 'file2ind.npy'), file2ind)
	np.save(os.path.join(datapath2, 'ind2file.npy'), ind2file)
	# print('---PARSING JSON FILES---')
	# # parse descriptor files (for time signature)
	# descrip_path = os.path.join(datapath, 'descriptor')
	# descrip_files = sorted(os.listdir(descrip_path))
	# savepath_d = os.path.join(datapath, 'descriptor_npy')
	# if not os.path.exists(savepath_d):
	# 	os.makedirs(savepath_d)
	# for i in range(len(descrip_files)):
	# 	f = open(os.path.join(descrip_path, descrip_files[i]))
	# 	meta_info = f.readlines()[5:7]
	# 	if len(meta_info) == 2:
	# 		data_ = [re.split(',|\n| ', m) for m in meta_info]
	# 		timesig = ""
	# 		for d in data_:
	# 			time = [t for t in d if len(t) is not 0]
	# 			timesig += time[0]+'/'
	# 		timesig = timesig[:-1]
	# 	else: timesig = 'NULL'
	# 	print(timesig, i)
	# 	ind = ind2string(i, 3)
	# 	# print(timesig, i)
	# 	np.save(os.path.join(savepath_d, '%s_poly_timesig.npy' % ind), timesig)
	# 	print('parsing %ith json files' % (i+1), end='\r')
	# print('parsed time signature files')

def ind2string(n, num=3):
    ind = str(n)
    rest = num - len(ind)
    new_ind = '0'*rest + ind
    return new_ind

class SaveBatches(object):
	def __init__(self, data_mode="note", batch_mode="reg"):
		self.data_mode = data_mode
		self.batch_mode = batch_mode
		'''
		* data_mode: "note" or "frame"
			- "note": parsing data as sequence of onehot vectors where each vector corresponds to each note
				- duration and pitch(mnn) data are parsed seperately, resulting two types of onehot vectors
			- "frame": parsing data as sequence of onehot vectors where each vector corresponds to each frame (piano roll)
		* batch_mode: "encdec" or "reg"
			- "encdec": slicing batches for encoder-decoder structure
			  	- where start and end timestep of encoder input are t and t+N respectively,
				- timesteps of decoder input start with t+N
				- timesteps of decoder output start with t+N+1
				- resulted batches are groups of three (enc_input, dec_input ,dec_output)
			- "reg": slicing batches for autoregressive(markov) structure
				- timesteps of input start with t
				- timesteps of output start with t+1
				- resulted batches are groups of two (input, output)
		'''
		self.datapath = '/home/rsy/Documents/MIREX/dataset/PPDD_mono_small'
		self.foil_path = os.path.join(self.datapath, 'cont_foil_npy')
		self.true_path = os.path.join(self.datapath, 'cont_true_npy')
		self.prime_path = os.path.join(self.datapath, 'prime_npy')
		self.descrip_path = os.path.join(self.datapath, 'descriptor_npy')
		self.all_path = [self.prime_path, self.true_path, self.foil_path]
		self.unit = [0.0, 0.08, 0.17, 0.25, 0.33, 0.42,
					 0.5, 0.58, 0.67, 0.75, 0.83, 0.92,
					 1.0, 1.08, 1.17, 1.25, 1.33, 1.42,
					 1.5, 1.58, 1.67, 1.75, 1.83, 1.92]
		self.unit2ind_short = dict()
		self.unit2ind_long = dict()

	def __call__(self):
		if self.data_mode == "note":
			print("parsing and saving in note mode...")
			self.note()
		elif self.data_mode == "frame":
			print("parsing and saving in frame mode...")
			self.frame()

	def note(self):
		savepath = os.path.join(self.datapath, 'note')
		for ui, u in enumerate(self.unit):
			self.unit2ind_long[u] = ui
		for ai, a in enumerate(self.all_path):
			filelist = sorted(glob(os.path.join(a, '*_raw.npy')))
			typename = a.split('/')[-1][:-4]
			# split data into one of train, val, test sets
			datalen = len(filelist)
			index = np.arange(datalen)
			split_rate = [0.7, 0.2, 0.1]
			train, val = [int(datalen * s) for s in split_rate[:-1]]
			test = train + val
			train_ind, val_ind, test_ind = index[:train], index[train:test], index[-test:]
			print('---Saving batches for %s folder---' % typename)
			for fi, f in enumerate(filelist):
				fileind = f.split('/')[-1][:-8]
				all_info = np.load(f)
				# make onehots for each note
				dur_onehot = np.zeros([len(all_info), len(self.unit)])
				mnn_onehot = np.zeros([len(all_info), 88])
				for ii, each_info in enumerate(all_info):
					pos, mnn, mpn, dur, chn = each_info
					# parse duration
					if ii < len(all_info)-1:
						dur = all_info[ii+1][0] - all_info[ii][0]
					elif ii == len(all_info)-1:
						dur = dur
					dur_ = np.round(dur, 2)
					if dur_ not in self.unit: # quantize into one of 12 units
						dist = [np.abs(dur_ - u) for u in self.unit]
						dur_ind = dist.index(np.min(dist))
					else: dur_ind = self.unit2ind_long[dur_]
					dur_onehot[ii][dur_ind] = 1
					# parse midi number into range 0-87
					mnn_ = int(mnn) - 21
					mnn_onehot[ii][mnn_] = 1
				# make input and output files and save them
				if ai == 0: # if prime:
					# split data into batches with shorter length
					dur_inp_batch, dur_oup_batch, \
					mnn_inp_batch, mnn_oup_batch = self.make_batches_note(dur_onehot, mnn_onehot)
					# save batches into tfrecord files for training
					if fi in train_ind:
						path = os.path.join(savepath, 'tfrecords')
						self.save_tfrecords_note(dur_inp_batch, dur_oup_batch,
						               			 mnn_inp_batch, mnn_oup_batch, path, fileind, 'train')
					elif fi in val_ind:
						path = os.path.join(savepath, 'tfrecords')
						self.save_tfrecords_note(dur_inp_batch, dur_oup_batch,
						               			 mnn_inp_batch, mnn_oup_batch, path, fileind, 'val')
					elif fi in test_ind:
						path = os.path.join(savepath, 'test')
						if not os.path.exists(path):
							os.makedirs(path)
						np.save(os.path.join(path,'%s_d_test_inp.npy' % fileind), dur_inp_batch)
						np.save(os.path.join(path,'%s_d_test_oup.npy' % fileind), dur_oup_batch)
						np.save(os.path.join(path,'%s_d_test_all.npy' % fileind), dur_onehot)
						np.save(os.path.join(path,'%s_m_test_inp.npy' % fileind), mnn_inp_batch)
						np.save(os.path.join(path,'%s_m_test_oup.npy' % fileind), mnn_oup_batch)
						np.save(os.path.join(path,'%s_m_test_all.npy' % fileind), mnn_onehot)
					print('saved %ith tfrecord set' % (fi+1), end='\r')
				elif ai > 0: # if cont
					path = os.path.join(savepath, 'GT', typename)
					if not os.path.exists(path):
						os.makedirs(path)
					np.save(os.path.join(path, '%s_d_GT.npy' % fileind), dur_onehot)
					np.save(os.path.join(path, '%s_m_GT.npy' % fileind), mnn_onehot)
					print('saved %ith GT files' % (fi+1), end='\r')

	def frame(self):
		savepath = os.path.join(self.datapath, 'frame')
		for ui, u in enumerate(self.unit[:12]):
			self.unit2ind_short[u] = ui
		frame = 12
		for ai, a in enumerate(self.all_path):
			filelist = sorted(glob(os.path.join(a, '*_raw.npy')))
			typename = a.split('/')[-1][:-4]
			datalen = len(filelist)
			index = np.arange(datalen)
			split_rate = [7, 2, 1]
			train, val, test = [int(datalen/np.sum(split_rate)) * s for s in split_rate]
			train_ind, val_ind, test_ind = index[:train], index[train:-test], index[-test:]
			print('---Saving batches for %s folder---' % typename)
			for fi, f in enumerate(filelist):
				fileind = f.split('/')[-1][:-8]
				all_info = np.load(f)
				last_meas = np.ceil(all_info[-1][0])
				if last_meas == all_info[-1][0]: # if last position is already int
					last_meas += 1
				# make piano roll
				piano_roll_note = np.zeros([int(last_meas)*frame,])
				for ii, info in enumerate(all_info):
					pos, mnn, mpn, dur, chn = info
					# parse measure position into frame position
					pos_int = int(pos)
					pos_unit = np.round(pos - pos_int, 2)
					if pos_unit not in self.unit[:12]: # quantize into one of 12 units
						dist = [np.abs(pos_unit - u) for u in self.unit[:12]]
						pos_ind = dist.index(np.min(dist))
					else: pos_ind = self.unit2ind_short[pos_unit]
					pos_ = pos_int * frame + pos_ind
					# decide start position
					if ai == 0 and ii == 0: # if prime
						first_pos = pos_int * frame
					elif ai > 0 and ii == 0: # if cont
						first_pos = pos_
					'''
					if prime, piano roll starts from the beginning of the first measure
					if cont, piano roll starts from the original position in the first measure
					(since cont is continuation of the prime)
					'''
					# parse duration into number of frames
					if ii < len(all_info)-1:
						dur = all_info[ii+1][0] - all_info[ii][0]
					elif ii == len(all_info)-1:
						dur = dur
					dur_int = int(dur)
					dur_unit = np.round(dur - dur_int, 2)
					if dur_unit not in self.unit[:12]: # quantize into one of 12 units
						dist = [np.abs(dur_unit - u) for u in self.unit[:12]]
						dur_ind = dist.index(np.min(dist))
					else: dur_ind = self.unit2ind_short[dur_unit]
					dur_ = dur_int * frame + dur_ind
					# parse midi number into range 0-87
					mnn_ = int(mnn) - 21
					piano_roll_note[pos_:pos_+dur_] = mnn_
				piano_roll = piano_roll_note[first_pos:]
				# make input and output files and save them
				if ai == 0: # if prime:
					# split data into batches with shorter length
					if self.batch_mode == "encdec": # for enc-dec structure
						inp1_batch, inp2_batch, oup_batch = self.make_batches_frame(piano_roll)
						# save batches into tfrecord files for training
						if fi in train_ind:
							path = os.path.join(savepath, 'tfrecords_%s' % self.batch_mode)
							if not os.path.exists(path):
								os.makedirs(path)
							self.save_tfrecords_frame(inp1_batch, inp2_batch, oup_batch, path, fileind, 'train')
						elif fi in val_ind:
							path = os.path.join(savepath, 'tfrecords_%s' % self.batch_mode)
							if not os.path.exists(path):
								os.makedirs(path)
							self.save_tfrecords_frame(inp1_batch, inp2_batch, oup_batch, path, fileind, 'val')
						elif fi in test_ind:
							path = os.path.join(savepath, 'test_%s' % self.batch_mode)
							if not os.path.exists(path):
								os.makedirs(path)
							np.save(os.path.join(path,'%s_test_inp1.npy' % fileind), inp1_batch)
							np.save(os.path.join(path,'%s_test_inp2.npy' % fileind), inp2_batch)
							np.save(os.path.join(path,'%s_test_oup.npy' % fileind), oup_batch)
							np.save(os.path.join(path,'%s_test_all.npy' % fileind), piano_roll)
					elif self.batch_mode == "reg": # for regression
						inp_batch, _, oup_batch = make_batches(piano_roll, mode=mode)
						# save batches into tfrecord files for training
						if fi in train_ind:
							path = os.path.join(savepath, 'tfrecords_%s' % self.batch_mode)
							if not os.path.exists(path):
								os.makedirs(path)
							save_tfrecords(inp_batch, None, oup_batch, path, fileind, 'train')
						elif fi in val_ind:
							path = os.path.join(savepath, 'tfrecords_%s' % self.batch_mode)
							if not os.path.exists(path):
								os.makedirs(path)
							save_tfrecords(inp_batch, None, oup_batch, path, fileind, 'val')
						elif fi in test_ind:
							path = os.path.join(savepath, 'test_%s' % self.batch_mode)
							if not os.path.exists(path):
								os.makedirs(path)
							np.save(os.path.join(path,'%s_test_inp.npy' % fileind), inp_batch)
							np.save(os.path.join(path,'%s_test_oup.npy' % fileind), oup_batch)
							np.save(os.path.join(path,'%s_test_all.npy' % fileind), piano_roll)
					print('saved %ith tfrecord set (mode: %s)' % (fi+1, self.batch_mode), end='\r')
				elif ai > 0: # if cont
					path = os.path.join(savepath, 'GT', typename)
					if not os.path.exists(path):
						os.makedirs(path)
					np.save(os.path.join(path, '%s_GT.npy' % fileind), piano_roll)
					print('saved %ith GT files' % (fi+1), end='\r')

	def make_batches_note(self, data1, data2):
		datalen = len(data1)
		maxlen = 50
		stride = 1
		data1_inp_batch = list()
		data1_oup_batch = list()
		data2_inp_batch = list()
		data2_oup_batch = list()
		for i in range(0, datalen, stride):
			data1_inp_batch.append(data1[i:i+maxlen])
			data1_oup_batch.append(data1[i+1:i+maxlen+1])
			data2_inp_batch.append(data2[i:i+maxlen])
			data2_oup_batch.append(data2[i+1:i+maxlen+1])
		data1_inp_batch = pad_sequences(data1_inp_batch, maxlen=maxlen, padding='post', dtype=np.float32)
		data1_oup_batch = pad_sequences(data1_oup_batch, maxlen=maxlen, padding='post', dtype=np.float32)
		data2_inp_batch = pad_sequences(data2_inp_batch, maxlen=maxlen, padding='post', dtype=np.float32)
		data2_oup_batch = pad_sequences(data2_oup_batch, maxlen=maxlen, padding='post', dtype=np.float32)
		return data1_inp_batch, data1_oup_batch, data2_inp_batch, data2_oup_batch

	def save_tfrecords_note(self, d_inp, d_oup, m_inp, m_oup, path, fileind, set_):
		# func for saving values into bytes
		def _bytes_feature(value):
			return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
		# save tfrecord files
		datalen = len(d_inp)
		path_ = os.path.join(path, '%s' % set_)
		if not os.path.exists(path_):
			os.makedirs(path_)
		filename = os.path.join(path_, '%s.tfrecords' % fileind)
		writer = tf.python_io.TFRecordWriter(filename)
		for i in range(datalen):
			d_inp_ = d_inp[i]
			d_oup_ = d_oup[i]
			m_inp_ = m_inp[i]
			m_oup_ = m_oup[i]
			feature_ = {'d_inp': _bytes_feature(tf.compat.as_bytes(d_inp_.tostring())),
						'd_oup': _bytes_feature(tf.compat.as_bytes(d_oup_.tostring())),
						'm_inp': _bytes_feature(tf.compat.as_bytes(m_inp_.tostring())),
						'm_oup': _bytes_feature(tf.compat.as_bytes(m_oup_.tostring()))}
			example_ = tf.train.Example(features=tf.train.Features(feature=feature_))
			writer.write(example_.SerializeToString())
		writer.close()
		sys.stdout.flush()

	def make_batches_frame(self, data):
		datalen = len(data)
		maxlen = 96 # 8 measures
		dim = 89
		stride = 6
		inp1_batch, inp2_batch, oup_batch = list(), list(), list()
		if self.batch_mode == "encdec":
			for i in range(0, datalen, stride):
				inp1_batch.append(data[i:i+maxlen])
				inp2_batch.append(data[i+maxlen-1:i+maxlen*2-1])
				oup_batch.append(data[i+maxlen:i+maxlen*2])
		elif self.batch_mode == "reg":
			for i in range(0, datalen, stride):
				inp1_batch.append(data[i:i+maxlen])
				oup_batch.append(data[i+1:i+maxlen+1])
		inp1_batch = pad_sequences(inp1_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
		oup_batch = pad_sequences(oup_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
		if mode == "encdec":
			inp2_batch = pad_sequences(inp2_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
		else: inp2_batch = None
		return inp1_batch, inp2_batch, oup_batch

	def save_tfrecords_frame(self, inp1, inp2, oup, path, fileind, set_):
		# func for saving values into bytes
		def _bytes_feature(value):
			return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
		# save tfrecord files
		datalen = len(inp1)
		path_ = os.path.join(path, '%s' % set_)
		if not os.path.exists(path_):
			os.makedirs(path_)
		filename = os.path.join(path_, '%s.tfrecords' % fileind)
		writer = tf.python_io.TFRecordWriter(filename)
		if self.batch_mode == "encdec":
			for i in range(datalen):
				inp1_ = inp1[i]
				inp2_ = inp2[i]
				oup_ = oup[i]
				feature_ = {'inp1': _bytes_feature(tf.compat.as_bytes(inp1_.tostring())),
							'inp2': _bytes_feature(tf.compat.as_bytes(inp2_.tostring())),
							'oup': _bytes_feature(tf.compat.as_bytes(oup_.tostring()))}
				example_ = tf.train.Example(features=tf.train.Features(feature=feature_))
				writer.write(example_.SerializeToString())
			writer.close()
			sys.stdout.flush()
		elif self.batch_mode == "reg":
			for i in range(datalen):
				inp1_ = inp1[i]
				oup_ = oup[i]
				feature_ = {'inp': _bytes_feature(tf.compat.as_bytes(inp1_.tostring())),
							'oup': _bytes_feature(tf.compat.as_bytes(oup_.tostring()))}
				example_ = tf.train.Example(features=tf.train.Features(feature=feature_))
				writer.write(example_.SerializeToString())
			writer.close()
			sys.stdout.flush()

# if __name__ == "__main__":
	# parse_info()
