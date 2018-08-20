import numpy as np
import os
import re
import sys
from glob import glob
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

'''
- This is a parsing code for "MIREX2018: Patterns of Prediction".
- Only 'monophonic' version works.
'''

def ind2string(n, num=3):
    ind = str(n)
    rest = num - len(ind)
    new_ind = '0'*rest + ind
    return new_ind

class ParseData(object):
	'''
	* data_mode: "note" or "frame"
		- "note": parsing data as sequence of onehot vectors 
				  where each vector corresponds to each note
			- duration and pitch(mnn) data are parsed seperately, 
			  resulting two types of onehot vectors
		- "frame": parsing data as sequence of onehot vectors 
		  where each vector corresponds to each frame (piano roll)
	* batch_mode: "encdec" or "reg"
		- "encdec": slicing batches for encoder-decoder structure
		  	- where start and end timestep of encoder input are t and t+N,
			- timesteps of decoder input start with t+N
			- timesteps of decoder output start with t+N+1
			- resulted batches are (enc_input, dec_input ,dec_output)
		- "reg": slicing batches for autoregressive(markov) structure
			- timesteps of input start with t
			- timesteps of output start with t+1
			- resulted batches are (input, output)
	'''
	def __init__(self, 
				 data_dir=None,
				 data_size=None, 
				 data_mode=None, 
				 batch_mode=None):
		self.data_dir = data_dir
		self.data_size = data_size 
		self.data_mode = data_mode
		self.batch_mode = batch_mode
		self.data_type = "prime"
		self.datapath = os.path.join(self.data_dir, 
			'PPDD_parsed_mono_%s/%s' % (data_size, self.data_type))
		self.unit = [0.0, 0.08, 0.17, 0.25, 0.33, 0.42,
					 0.5, 0.58, 0.67, 0.75, 0.83, 0.92]
		self.dur_dim = 96
		self.mnn_dim = 88
		self.inp_dim = 89
		self.timestep_frame = 96
		self.timestep_note = 20

	def __call__(self, only_batch=False):
		if self.data_mode == "note":
			print("parsing in note mode...")
			if only_batch == False:
				self.parse_raw()
				self.save_notes()
			elif only_batch == True:
				self.save_notes()
		elif self.data_mode == "frame":
			print("parsing in frame mode...")
			if only_batch == False:
				self.parse_raw()
				self.save_frames()
			elif only_batch == True:
				self.save_frames()

	def parse_raw(self, path=None):
		rawpath = os.path.join(self.data_dir,
			'PPDD-Jul2018_sym_mono_%s/PPDD-Jul2018_sym_mono_%s/%s_csv' % \
			(self.data_size, self.data_size, self.data_type))
		savepath = os.path.join(self.datapath, "raw")
		# if path does not exist, create one
		if not os.path.exists(savepath): 
			os.makedirs(savepath) 
		datalen_list = list()
		pos_list = dict()
		mnn_list = dict()
		dur_list = dict() 
		itv_list = dict()
		file2ind, ind2file = dict(), dict()
		print('---PARSING CSV FILES---')
		# parse prime files
		filelist = sorted(glob(os.path.join(rawpath, "*.csv")))
		for fi, each_file in enumerate(filelist): # brings each file 
			ind = ind2string(fi, 3) # make string index ("000", "001" etc.)
			filename = each_file.split('/')[-1]
			file2ind[filename] = ind
			ind2file[ind] = filename
			f = open(os.path.join(each_file), 'r') # open file
			data = f.readlines() # read file
			each_file_data = list()
			datalen = 0
			last_mnn = 0
			for d, each_data in enumerate(data): # brings each note
				splited_ = re.split(',|\n', each_data)[:-1] # split by ',' and '\n'
				s = [np.round(float(ss),2) for ss in splited_] # str to float
				pos, mnn, dur = s[0], s[1], s[3]
				try: # get dictionaries for histrogram
					pos_list[pos-int(pos)] += 1 
					mnn_list[mnn] += 1
					dur_list[dur] += 1 
					itv_list[mnn-last_mnn] += 1
				except:
					pos_list[pos-int(pos)] = 1
					mnn_list[mnn] = 1
					dur_list[dur] = 1
					itv_list[mnn-last_mnn] = 1
				each_file_data.append(s)
				datalen += 1
				last_mnn = mnn
			datalen_list.append(datalen) # datalen list to figure out how long
			np.save(os.path.join(savepath, '%s_mono_raw.npy' % ind), each_file_data)
			f.close()
			print('parsing %ith csv files' % (fi+1), end='\r')
		print()
		# save npy files
		np.save(os.path.join(savepath, 'datalen_list.npy'), datalen_list)
		np.save(os.path.join(savepath, 'pos_list.npy'), pos_list)
		np.save(os.path.join(savepath, 'mnn_list.npy'), mnn_list)
		np.save(os.path.join(savepath, 'dur_list.npy'), dur_list)
		np.save(os.path.join(savepath, 'itv_list.npy'), itv_list)
		np.save(os.path.join(savepath, 'file2ind.npy'), file2ind)
		np.save(os.path.join(savepath, 'ind2file.npy'), ind2file)

	def save_notes(self):
		savepath = os.path.join(self.datapath, 'note')
		tf_savepath = os.path.join(savepath, 'tfrecords')
		test_path = os.path.join(savepath, 'test')
		if not os.path.exists(tf_savepath):
			os.makedirs(os.path.join(tf_savepath, "train"))
			os.makedirs(os.path.join(tf_savepath, "val"))
		if not os.path.exists(test_path):
			os.makedirs(test_path)
		unit_long = list()
		unit2ind_long = dict()
		for i in range(8): # max IOI is 8 quarter beats long
			for s in self.unit:
				unit_long.append(s+i)
		for ui, u in enumerate(unit_long):
			unit2ind_long[u] = ui
		dur_hist = dict()
		mnn_hist = dict()
		for s in ['train', 'val', 'test']:
			dur_hist[s] = dict()
			mnn_hist[s] = dict()
		filelist = sorted(glob(os.path.join(self.datapath, 'raw/*_raw.npy')))
		# split data into one of train, val, test sets
		datalen = len(filelist)
		index = np.arange(datalen)
		split_rate = [0.7, 0.9, 1.0]
		train, val, test = [int(datalen*s) for s in split_rate]
		train_ind, val_ind, test_ind = index[:train], index[train:val], index[val:test]
		print('---SAVING BATCHES---')
		for fi, f in enumerate(filelist):
			fileind = f.split('/')[-1][:-8]
			all_notes = np.load(f)
			# make onehots for each note
			dur_onehot = np.zeros([len(all_notes), self.dur_dim])
			mnn_onehot = np.zeros([len(all_notes), self.mnn_dim])
			for ni, each_note in enumerate(all_notes):
				pos, mnn, mpn, dur, chn = each_note
				# parse duration
				if ni < len(all_notes)-1:
					dur = all_notes[ni+1][0] - all_notes[ni][0]
				elif ni == len(all_notes)-1:
					dur = dur
				dur_ = np.round(dur, 2)
				if dur_ not in self.unit: # quantize into one of 12 units
					dist = [np.abs(dur_ - u) for u in unit_long]
					dur_ind = dist.index(np.min(dist))
				else: dur_ind = unit2ind_long[dur_]
				dur_onehot[ni][dur_ind] = 1
				# parse midi number into range 0-87
				mnn_ind = int(mnn) - 21
				mnn_onehot[ni][mnn_ind] = 1
				# for histogram
				if fi in train_ind:
					try:
						dur_hist['train'][dur_ind] += 1
						mnn_hist['train'][mnn_ind] += 1
					except KeyError:
						dur_hist['train'][dur_ind] = 1
						mnn_hist['train'][mnn_ind] = 1
				elif fi in val_ind:
					try:
						dur_hist['val'][dur_ind] += 1
						mnn_hist['val'][mnn_ind] += 1
					except KeyError:
						dur_hist['val'][dur_ind] = 1
						mnn_hist['val'][mnn_ind] = 1
				elif fi in test_ind:
					try:
						dur_hist['test'][dur_ind] += 1
						mnn_hist['test'][mnn_ind] += 1
					except KeyError:
						dur_hist['test'][dur_ind] = 1
						mnn_hist['test'][mnn_ind] = 1
			# make input and output files and save them
			dur_inp_batch, dur_oup_batch, \
			mnn_inp_batch, mnn_oup_batch = self.make_batches_note(dur_onehot, mnn_onehot)
			# save batches into tfrecord files for training
			if fi in train_ind:
				self.save_tfrecords_note(dur_inp_batch, dur_oup_batch,
										 mnn_inp_batch, mnn_oup_batch, 
										 tf_savepath, fileind, 'train')
			elif fi in val_ind:
				self.save_tfrecords_note(dur_inp_batch, dur_oup_batch,
										 mnn_inp_batch, mnn_oup_batch, 
										 tf_savepath, fileind, 'val')
			elif fi in test_ind:		
				np.save(os.path.join(test_path,'%s_d_test_inp.npy' % fileind), dur_inp_batch)
				np.save(os.path.join(test_path,'%s_d_test_oup.npy' % fileind), dur_oup_batch)
				np.save(os.path.join(test_path,'%s_d_test_all.npy' % fileind), dur_onehot)
				np.save(os.path.join(test_path,'%s_m_test_inp.npy' % fileind), mnn_inp_batch)
				np.save(os.path.join(test_path,'%s_m_test_oup.npy' % fileind), mnn_oup_batch)
				np.save(os.path.join(test_path,'%s_m_test_all.npy' % fileind), mnn_onehot)
			print('saved %ith tfrecord sets & test set' % (fi+1), end='\r')
		print()
		np.save(os.path.join(savepath,'dur_hist_note.npy'), dur_hist)
		np.save(os.path.join(savepath,'mnn_hist_note.npy'), mnn_hist)
		print("---DONE---")

	def save_frames(self):
		savepath = os.path.join(self.datapath, 'frame')
		tf_savepath = os.path.join(savepath, '%s/tfrecords' % self.batch_mode)
		test_path = os.path.join(savepath, '%s/test' % self.batch_mode)
		if not os.path.exists(tf_savepath):
			os.makedirs(os.path.join(tf_savepath, "train"))
			os.makedirs(os.path.join(tf_savepath, "val"))
		if not os.path.exists(test_path):
			os.makedirs(test_path)
		unit2ind_short = dict()	
		for ui, u in enumerate(self.unit):
			unit2ind_short[u] = ui
		mnn_hist = dict()
		for s in ['train', 'val', 'test']:
			mnn_hist[s] = dict()
		frame = 12
		filelist = sorted(glob(os.path.join(self.datapath, 'raw/*_raw.npy')))
		datalen = len(filelist)
		index = np.arange(datalen)
		split_rate = [0.7, 0.9, 1.0]
		train, val, test = [int(datalen*s) for s in split_rate]
		train_ind, val_ind, test_ind = index[:train], index[train:val], index[val:test]
		print('---SAVING BATCHES---')
		for fi, f in enumerate(filelist):
			fileind = f.split('/')[-1][:-8]
			all_info = np.load(f)
			if int(all_info[-1][0]) == all_info[-1][0]:
				last_quar = all_info[-1][0] + 1
			else: last_quar = np.ceil(all_info[-1][0])
			# make piano roll
			piano_roll = np.full([int(last_quar)*frame,], fill_value=88)
			for ii, info in enumerate(all_info):
				pos, mnn, mpn, dur, chn = info
				# parse measure position into frame position
				pos_int = int(pos)
				pos_unit = np.round(pos - pos_int, 2)
				if pos_unit not in self.unit: # quantize into one of 12 units
					dist = [np.abs(pos_unit - u) for u in self.unit]
					pos_ind = dist.index(np.min(dist))
				else: pos_ind = unit2ind_short[pos_unit]
				pos_ = pos_int * frame + pos_ind
				# start position
				if ii == 0:
					first_pos = pos_int * frame
				# parse duration into number of frames
				if ii < len(all_info)-1:
					dur_ = all_info[ii+1][0] - all_info[ii][0]
					if dur > dur_:
						dur = dur_
					else: dur = dur
				elif ii == len(all_info)-1:
					dur = dur
				dur_int = int(dur)
				dur_unit = np.round(dur - dur_int, 2)
				if dur_unit not in self.unit: # quantize into one of 12 units
					dist = [np.abs(dur_unit - u) for u in self.unit]
					dur_ind = dist.index(np.min(dist))
				else: dur_ind = unit2ind_short[dur_unit]
				dur_fr = dur_int * frame + dur_ind
				# parse midi number into range 0-87
				mnn_fr = int(mnn) - 21
				piano_roll[pos_:pos_+dur_fr] = mnn_fr

			piano_roll = piano_roll[first_pos:]
			# for histogram
			for m in piano_roll:
				if fi in train_ind:
					try:
						mnn_hist['train'][m] += 1
					except KeyError:
						mnn_hist['train'][m] = 1
				elif fi in val_ind:
					try:
						mnn_hist['val'][m] += 1
					except KeyError:
						mnn_hist['val'][m] = 1
				elif fi in test_ind:
					try:
						mnn_hist['test'][m] += 1
					except KeyError:
						mnn_hist['test'][m] = 1
			# make input and output files and save them
			# for enc-dec structure
			# if self.batch_mode == "encdec": 
			# 	inp1_batch, inp2_batch, oup_batch = self.make_batches_frame(piano_roll)
			# 	# save batches into tfrecord files 
			# 	if fi in train_ind:
			# 		self.save_tfrecords_frame([inp1_batch, inp2_batch, oup_batch], 
			# 								  tf_savepath, fileind, 'train')
			# 	elif fi in val_ind:
			# 		self.save_tfrecords_frame([inp1_batch, inp2_batch, oup_batch], 
			# 								  tf_savepath, fileind, 'val')
			# 	elif fi in test_ind:
			# 		np.save(os.path.join(test_path,'%s_test_inp1.npy' % fileind), inp1_batch)
			# 		np.save(os.path.join(test_path,'%s_test_inp2.npy' % fileind), inp2_batch)
			# 		np.save(os.path.join(test_path,'%s_test_oup.npy' % fileind), oup_batch)
			# 		np.save(os.path.join(test_path,'%s_test_all.npy' % fileind), piano_roll)
			# # for autoregressive structure
			# elif self.batch_mode == "reg": 
			# 	inp_batch, oup_batch = self.make_batches_frame(piano_roll)
			# 	# save batches into tfrecord files 
			# 	if fi in train_ind:
			# 		self.save_tfrecords_frame([inp_batch, oup_batch], 
			# 								  tf_savepath, fileind, 'train')
			# 	elif fi in val_ind:
			# 		self.save_tfrecords_frame([inp_batch, oup_batch], 
			# 								  tf_savepath, fileind, 'val')
			# 	elif fi in test_ind:
			# 		np.save(os.path.join(test_path,'%s_test_inp.npy' % fileind), inp_batch)
			# 		np.save(os.path.join(test_path,'%s_test_oup.npy' % fileind), oup_batch)
			# 		np.save(os.path.join(test_path,'%s_test_all.npy' % fileind), piano_roll)
			print('saved %ith tfrecord sets & test set (mode: %s)' \
				% (fi+1, self.batch_mode), end='\r')
		print()
		np.save(os.path.join(savepath,'mnn_hist_frame.npy'), mnn_hist)
		print("---DONE---")

	def make_batches_note(self, data1, data2):
		datalen = len(data1)
		maxlen = self.timestep_note
		stride = 1
		data1_inp_batch = list()
		data1_oup_batch = list()
		data2_inp_batch = list()
		data2_oup_batch = list()
		for i in range(0, datalen-maxlen-1, stride):
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
		filename = os.path.join(path, '%s/%s.tfrecords' % (set_, fileind))
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
		maxlen = self.timestep_frame # 1 measures
		dim = self.inp_dim
		stride = 1
		if self.batch_mode == "encdec":
			inp1_batch, inp2_batch, oup_batch = list(), list(), list()
			for i in range(0, datalen-maxlen*2, stride):
				inp1_batch.append(data[i:i+maxlen])
				inp2_batch.append(data[i+maxlen-1:i+maxlen*2-1])
				oup_batch.append(data[i+maxlen:i+maxlen*2])
			inp1_batch = pad_sequences(inp1_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
			inp2_batch = pad_sequences(inp2_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
			oup_batch = pad_sequences(oup_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
			return [inp1_batch, inp2_batch, oup_batch]
		elif self.batch_mode == "reg":
			inp_batch, oup_batch = list(), list()
			for i in range(0, datalen-maxlen-1, stride):
				inp_batch.append(data[i:i+maxlen])
				oup_batch.append(data[i+1:i+maxlen+1])
			inp_batch = pad_sequences(inp_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
			oup_batch = pad_sequences(oup_batch, maxlen=maxlen, padding='post', dtype=np.int32, value=88)
			return [inp_batch, oup_batch]

	def save_tfrecords_frame(self, data, path, fileind, set_):
		# func for saving values into bytes
		def _bytes_feature(value):
			return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
		# save tfrecord files
		datalen = len(data[0])
		filename = os.path.join(path, '%s/%s.tfrecords' % (set_, fileind))
		writer = tf.python_io.TFRecordWriter(filename)
		if self.batch_mode == "encdec":
			for i in range(datalen):
				inp1_ = data[0][i]
				inp2_ = data[1][i]
				oup_ = data[2][i]
				feature_ = {'inp1': _bytes_feature(tf.compat.as_bytes(inp1_.tostring())),
							'inp2': _bytes_feature(tf.compat.as_bytes(inp2_.tostring())),
							'oup': _bytes_feature(tf.compat.as_bytes(oup_.tostring()))}
				example_ = tf.train.Example(features=tf.train.Features(feature=feature_))
				writer.write(example_.SerializeToString())
			writer.close()
			sys.stdout.flush()
		elif self.batch_mode == "reg":
			for i in range(datalen):
				inp_ = data[0][i]
				oup_ = data[1][i]
				feature_ = {'inp': _bytes_feature(tf.compat.as_bytes(inp_.tostring())),
							'oup': _bytes_feature(tf.compat.as_bytes(oup_.tostring()))}
				example_ = tf.train.Example(features=tf.train.Features(feature=feature_))
				writer.write(example_.SerializeToString())
			writer.close()
			sys.stdout.flush()

class ParseTfrecords(object):
    def __init__(self,
                 train_path=None,
                 val_path=None,
                 num_epoch=None,
                 batch_size1=None,
                 batch_size2=None,
                 data_mode=None,
                 batch_mode=None):
        self.train_path = train_path
        self.val_path = val_path
        self.data_mode = data_mode
        self.batch_mode = batch_mode
        self.num_epoch = num_epoch
        self.batch_size1 = batch_size1
        self.batch_size2 = batch_size2
        self.timestep_note = 20
        self.timestep_frame = 96
        self.dur_dim = 96
        self.mnn_dim = 88
        self.inp_dim = 89

    def __call__(self):
        if self.data_mode == "frame":
            train_data = self.parse_tfrecord_frame(
            	path=self.train_path, shuffle=True, setname="train")
            val_data = self.parse_tfrecord_frame(
            	path=self.val_path, shuffle=False, setname="val")
        if self.data_mode == "note":
            train_data = self.parse_tfrecord_note(
            	path=self.train_path, shuffle=True, setname="train")
            val_data = self.parse_tfrecord_note(
            	path=self.val_path, shuffle=False, setname="val")
        return train_data, val_data

    def parse_tfrecord_note(self, path, shuffle, setname):
        timesteps = self.timestep_note
        dur_dim, mnn_dim = self.dur_dim, self.mnn_dim
        if setname not in ['train', 'val']:
            raise ValueError
        # filenames are shuffled(default) every epoch:
        filename_queue = tf.train.string_input_producer(
            path, num_epochs=self.num_epoch, shuffle=shuffle)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # parse serialized files
        feature = {'d_inp': tf.FixedLenFeature([], tf.string),
                   'd_oup': tf.FixedLenFeature([], tf.string),
                   'm_inp': tf.FixedLenFeature([], tf.string),
                   'm_oup': tf.FixedLenFeature([], tf.string)}
        features = tf.parse_single_example(serialized_example, features=feature)
        # byte to int32
        d_inp = tf.decode_raw(features['d_inp'], tf.float32)
        d_oup = tf.decode_raw(features['d_oup'], tf.float32)
        m_inp = tf.decode_raw(features['m_inp'], tf.float32)
        m_oup = tf.decode_raw(features['m_oup'], tf.float32)
        d_inp = tf.reshape(d_inp, (timesteps, dur_dim))
        d_oup = tf.reshape(d_oup, (timesteps, dur_dim))
        m_inp = tf.reshape(m_inp, (timesteps, mnn_dim))
        m_oup = tf.reshape(m_oup, (timesteps, mnn_dim))
        if setname == 'train':
            di, do, mi, mo = tf.train.shuffle_batch([d_inp, d_oup, m_inp, m_oup],
                                                    batch_size=self.batch_size1,
                                                    capacity=1000+3*self.batch_size1,
                                                    num_threads=4,
                                                    min_after_dequeue=1000,
                                                    allow_smaller_final_batch=True)
        elif setname == 'val':
            di, do, mi, mo = tf.train.batch([d_inp, d_oup, m_inp, m_oup],
                                            batch_size=self.batch_size2,
                                            capacity=1000+3*self.batch_size2,
                                            allow_smaller_final_batch=True,
                                            num_threads=1)
        return [di, do, mi, mo]

    def parse_tfrecord_frame(self, path, shuffle, setname):
        timesteps = self.timestep_frame
        dim = self.inp_dim
        if setname not in ['train', 'val']:
            raise ValueError
        # filenames are shuffled(default) every epoch:
        filename_queue = tf.train.string_input_producer(
            path, num_epochs=self.num_epoch, shuffle=shuffle)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # parse serialized files
        if self.batch_mode == "encdec":
            feature = {'inp1': tf.FixedLenFeature([], tf.string),
                       'inp2': tf.FixedLenFeature([], tf.string),
                       'oup': tf.FixedLenFeature([], tf.string)}
            features = tf.parse_single_example(serialized_example, features=feature)
            # byte to int32
            inp1 = tf.decode_raw(features['inp1'], tf.int32)
            inp2 = tf.decode_raw(features['inp2'], tf.int32)
            oup = tf.decode_raw(features['oup'], tf.int32)
            inp1 = tf.reshape(inp1, (timesteps,))
            inp2 = tf.reshape(inp2, (timesteps,))
            oup = tf.reshape(oup, (timesteps,))
            if setname == 'train':
                i1, i2, o = tf.train.shuffle_batch([inp1, inp2, oup],
                                                   batch_size=self.batch_size1,
                                                   capacity=1000+3*self.batch_size1,
                                                   num_threads=4,
                                                   min_after_dequeue=1000,
                                                   allow_smaller_final_batch=True)
            elif setname == 'val':
                i1, i2, o = tf.train.batch([inp1, inp2, oup],
                                           batch_size=self.batch_size2,
                                           capacity=1000+3*self.batch_size2,
                                           allow_smaller_final_batch=True,
                                           num_threads=1)
            return [i1, i2, o]
        elif self.batch_mode == "reg":
            feature = {'inp': tf.FixedLenFeature([], tf.string),
                       'oup': tf.FixedLenFeature([], tf.string)}
            features = tf.parse_single_example(serialized_example, features=feature)
            # byte to int32
            inp = tf.decode_raw(features['inp'], tf.int32)
            oup = tf.decode_raw(features['oup'], tf.int32)
            inp = tf.reshape(inp, (timesteps,))
            oup = tf.reshape(oup, (timesteps,))
            if setname == 'train':
                i, o = tf.train.shuffle_batch([inp, oup],
                                              batch_size=self.batch_size1,
                                              capacity=1000+3*self.batch_size1,
                                              num_threads=4,
                                              min_after_dequeue=1000,
                                              allow_smaller_final_batch=True)
            elif setname == 'val':
                i, o = tf.train.batch([inp, oup],
                                      batch_size=self.batch_size2,
                                      capacity=1000+3*self.batch_size2,
                                      allow_smaller_final_batch=True,
                                      num_threads=1)
            return [i, o]


if __name__ == "__main__":
	if len(sys.argv) == 1:
		raise SyntaxError("Type your data directory!")
	elif len(sys.argv) == 2:
		raise SyntaxError("Type data mode (ex. note or frame)")
	elif len(sys.argv) == 3:
		data_dir = str(sys.argv[1])
		data_mode = str(sys.argv[2])
		if sys.argv[2] == "note":
			batch_mode = None
		elif sys.argv[2] == "frame":
			raise SyntaxError("Type batch mode (ex. encdec or reg)")
		else: 
			raise SyntaxError("Type batch mode (ex. encdec or reg)")
	elif len(sys.argv) == 4:
		data_dir = str(sys.argv[1])
		data_mode = str(sys.argv[2])
		batch_mode = str(sys.argv[3])
	parse_data = ParseData(data_dir=data_dir, 
						   data_size="small", 
						   data_mode=data_mode, 
						   batch_mode=batch_mode)
	parse_data(only_batch=False)
