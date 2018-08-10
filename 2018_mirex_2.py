import numpy as np
import os  
import re
import sys
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from glob import glob 


"""
   "This is a code for parsing symbolic 'monophonic','polyphonic' data 
    for "MIREX2018: Patterns of Prediction".
""" 


def parse_info():
	datapath1 = '/home/sarah/Documents/mirex_dataset/2018_mirex_original/PPDD-Jul2018_sym_mono_small/PPDD-Jul2018_sym_mono_small'
	datapath2 = '/home/sarah/Documents/mirex_dataset/2018_mirex_cleaning/sym_mono_small'
	foil_path = os.path.join(datapath, 'cont_foil_csv')
	true_path = os.path.join(datapath, 'cont_true_csv')
	prime_path = os.path.join(datapath, 'prime_csv')
 	# descrip_path = os.path.join(datapath, 'descriptor')
	all_path = [prime_path, true_path, foil_path]
	datalen_list = dict()
	mnn_list = dict()
	pos_list = dict()
	dur_list = dict()
	chn_list = dict()
	file2ind = dict()
	ind2file = dict()
	print('---PARSING CSV FILES---')

	## parse prime, true and foil files
	for path in all_path: ## each type of path among [prime, true, foil]
		typename = all_path[a].split('/')[-1][:-4]
		filelist = sorted(glob (os.path.join(path, "*.csv"))) ## brings all file lists
		datalen_list[typename] = list()
		mnn_list[typename] = dict()
		pos_list[typename] = dict()
		dur_list[typename] = dict()
		chn_list[typename] = dict()
		savepath = os.path.join(datapath2, '%s_npy' % typename)
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		for fi, each_file in enumerate(filelist): ## brings each file list 
			ind = ind2string(fi,3)
			filename = each_file.split('/')[-1]
			file2ind[filename] = ind
			ind2file[ind] = filename
			f = open(os.path.join(path, filename),'r') ## open file 
			data = f.readlines() ## read file 	
			each_file_data = list()
			datalen = 0
			for each_data in data: ## brings each line in csv data
				splited_ = re.split(',|\n', each_data) ## split by ',' and '\n'
				# splited_data = splited_[:-1] # without "" at the end 
				# each_file_data.append(splited_data)
				# datalen += 1
				s = splited_[:-1] ## without "" at the end
				float_s = [float(ss) for ss in s]
				try:
					mnn_list[typename][float_s[1]] += 1 ## MNN list for histogram
					dur_list[typename][np.round(float_s[3],2)] += 1 ## dur list for histogram
					pos_list[typename][np.round(float_s[0] -int(float_s[0]),2)] +=1 ## pos list for histogram
					chn_list[typename][float_s[4]] += 1 ## chn list for histrogram
				except:
					mnn_list[typename][float_s[1]] = 1
					dur_list[typename][np.round(float_s[3],2)] = 1
					pos_list[typename][np.round(float_s[0] -int(float_s[0]),2)] = 1
					chn_list[typename][float_s[4]] = 1
				each_file_data.append(float_s)
				datalen += 1
			datalen_list[typename].append(datalen) ## datalen list to figure out how long
			np.save(os.path.join(savepath, '%s_mono_raw.npy' % ind), each_file_data)
			f.close()
			print('parsing %ith csv files' % (fi+1), end='\r')
		# np.save(os.path.join(savepath, 'datalen_list.npy'), datalen_list)
		print('parsed %s csv files' % typename)

	## save npy files
	np.save(os.path.join(datapath2, 'datalen_list.npy'), datalen_list)
	np.save(os.path.join(datapath2, 'mnn_list.npy'), mnn_list)
	np.save(os.path.join(datapath2, 'dur_list.npy'), dur_list)
	np.save(os.path.join(datapath2, 'pos_list.npy'), pos_list)
	np.save(os.path.join(datapath2, 'chn_list.npy'), chn_list)
	np.save(os.path.join(datapath2, 'file2ind.npy'), file2ind)
	np.save(os.path.join(datapath2, 'ind2file.npy'), ind2file)
	print ('--- PARSTNG JSON FILES ---')

	##  parse descriptor files (for time signature)
	# descrip_path =  os.path.join(datapath, 'descriptor')
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


def save_batches():
	datapath = '/home/sarah/Documents/mirex_dataset/2018_mirex_cleaning/sym_mono_small'
	foil_path = os.path.join(datapath, 'cont_foil_npy')
	true_path = os.path.join(datapath, 'cont_true_npy')
	prime_path = os.path.join(datapath, 'prime_npy')
	descrip_path = os.path.join(datapath, 'descriptor_npy')
	all_path = [prime_path, true_path, foil_path]
	unit = [0.0, 0.08, 0.17, 0.25, 0.33, 0.42, 
			0.5, 0.58, 0.67, 0.75, 0.83, 0.92]
	unit2ind = dict()
	for ui, u in enumerate(unit):
		unit2ind[u] = ui 
	frame = 12
	print ('--- SAVING BATCHES ---')

	for ai, a in enumerate(all_path):
		filelist = sorted(glob(os.path.join(a, '*_raw.npy')))
		typename = a.split('/')[-1][:-4]
		datalen = len(filelist)
		index = np.arange(datalen)
		split_rate = [7, 2, 1]
		train, val, test = [int(datalen/np.sum(split_rate)) * s for s in split_rate]
		train_ind, val_ind, test_ind = index[:train], index[train:-test], index[-test:]
		print ('---Saving batches for %s folder---' % typename)
		for fi, f in enumerate(filelist):
			fileind = f.split('/')[-1][:-8]
			all_info = np.load(f) 
			last_meas = np.ceil(all_info[-1][0])
			if last_meas == all_info[-1][0]: ## if last posotion is already int 
				last_meas += 1
			# num_meas = last_meas - first_meas
			## make piano roll 
			piano_roll_note = np.zeros([int(last_meas)*frame, 88])
			piano_roll_rest = np.one([int(last_meas)*frame, 1])
			for ii , info in enumerate(all_info):
				pos, mnn, mpn, dur, chn = info
				## parse measure position into frame position
				pos_int = int(pos)
				pos_unit = pos - pos_int
				if pos_unit not in unit: ## quantize into one of 12 units
					dist = [pos_unit - u for u in unit]
					pos_ind = dist.index(np.min(dist))
				else: pos_ind = unit2ind[pos_unit]
				pos_ = pos_int * frame + pos_ind 
		
				'''
				if prime, piano roll starts from the beginning of the first measure
				if cont, piano roll starts from the original position in the first measure
				(since cont is continuation of the prime) 
				'''
				## decide start position 
				if ai == 0 and ii == 0: # if prime 
					first_pos = pos_int * frame
				elif ai > 0 and ii == 0: # if cont 
					first_pos = pos_
				## parse duration into number of frames
				dur_int = int(dur)
				dur_unit = dur - dur_int
				if dur_unit not in unit: # quantize into one of 12 units
					dist = [dur_unit - u for u in unit]
					dur_ind = dist.index(np.min(dist))
				else: dur_ind = unit2ind[dur_unit]
				dur_ = dur_int * frame + dur_ind
				# parse midi number into range 0-87
				mnn_ = int(mnn) - 21
				piano_roll_note[pos_:pos_+dur_][:,mnn_] = 1
				piano_roll_rest[pos_:pos +dur_][:, 0] = 0
				piano_roll = np.concatenate ([piano_roll_note, piano_roll_rest], axis = -1)
			piano_roll = piano_roll[first_pos:]

			## make input and output files and save them
			if ai == 0: ## if prime:
				inp = piano_roll[:-1]
				oup = piano_roll[1:]
				## save batches info tfrecord files for trainig
				if fi in train_ind:
					save_tfrecords(inp_batches, oup_batches, fileind, 'train')
				elif fi in val_ind:
					save_tfrecords(inp_batches, oup_batches, fileind, 'val')
				elif fi in test_ind:
					save_tfrecords(inp_batches, oup_batches, fileind, 'test')
				print('saved %ith tfrecord set' % (fi+1), end = '\r')
			elif ai > 0: ## if cont
				savepath = os.path.join(datapath, 'piano_roll', typename)
				if not os.path.exists(savepath):
					os.makedirs(savepath)
				np.save(os.path.join(savepath, '%s_piano_roll.npy' % fileind), piano_roll)
				print('saved %ith piano roll files' % (fi+1), end = '\r')

	def make_batches(data):
		maxlen = 48 ## 4 measures
		datalen = len(data)
		stride = 6
		batches = list()
		for i in range(0, datalen, stride):
			batches.append(data[i:i + maxlen])
		batches = pad_sequences(batches, padding = 'post', dtype = np.float32)
		return batches

	def save_tfrecords(inp, oyp, fileind, set_):
		## func for saveing values intp bytes
		def _bytes_feature(value):
			return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))
			##func for saving values into int64
		# def _int64_feature(value):
			# return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
		## save tfrecord files

		datalen = len(inp)
		savepath = '/home/sarah/Documents/mirex_dataset/2018_mirex_cleaning/sym_mono_small/tfrecords/%s' % set_
		if not os.path.exists(savepath):os.makedirs(savepath)
		filename = os.path.join(savepath, '%s.tfrecords' % fileind)
		writer = tf.python_io.TFRecordWriter(filename)
		for i in range(datalen): ## train set
			inp_ = inp[i]
			oup_ = oup[i]
			feature_ = {'inp': _bytes_feature(tf.compat.as_bytes(inp_.tostring())),\
						'oup': _bytes_feature(tf.compat.as_bytes(oup_.tostring()))}
			example_ = tf.train.Example(features=tf.train.Feature(feature=feature_))
			writer.write(example_.SerializeToString())
		writer.close()
		sys.stdout.flush()


if __name__ == "__main__":
	parse_info()
    