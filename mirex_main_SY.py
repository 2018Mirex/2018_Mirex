#! /usr/bin/env python
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

from mirex_model_SY import *
from mirex_data_SY import *

'''
- This is a code for a main experiment for "MIREX2018: Patterns of Prediction".
'''

class Graph(object):
    def __init__(self,
                 data=None,
                 data_mode=None,
                 batch_mode=None,
                 model_mode=None):
        # INPUTS
        if data_mode == "note":
            # attributes
            self.timestep = 20
            self.dur_dim = 96
            self.mnn_dim = 88
            self.emb_dim = 32
            # inputs
            inpD, trueD, inpM, trueM = data
            self.phase = tf.placeholder(dtype=tf.bool, shape=None, name='phase')
            self.inpD = tf.placeholder_with_default(inpD, shape=[None, None, self.dur_dim], name='dur_in')
            self.inpM = tf.placeholder_with_default(inpM, shape=[None, None, self.mnn_dim], name='mnn_in')
            self.trueD = tf.placeholder_with_default(trueD, shape=[None, None, self.dur_dim], name='dur_out')
            self.trueM = tf.placeholder_with_default(trueM, shape=[None, None, self.mnn_dim], name='mnn_out')
        elif data_mode == "frame":
            # attributes
            self.inp_dim = 89
            self.emb_dim = 32
            # inputs
            self.phase = tf.placeholder(dtype=tf.bool, shape=None, name='phase')
            if batch_mode == "encdec":
                inp1, inp2, oup = data
                self.inpM_enc = tf.placeholder_with_default(inp1, shape=[None, None], name='enc_in')
                self.inpM_dec = tf.placeholder_with_default(inp2, shape=[None, None], name='dec_in')
                self.trueM = tf.placeholder_with_default(oup, shape=[None, None], name='dec_out')
            if batch_mode == "reg":
                inp, oup = data
                self.inpM = tf.placeholder_with_default(inp, shape=[None, None], name='enc_in')
                self.trueM = tf.placeholder_with_default(oup, shape=[None, None], name='dec_out')
                # self.inpM = tf.placeholder(tf.int32, (None, None), name='enc_in')
                # self.trueM = tf.placeholder(tf.int32, (None, None), name='dec_out')

        # MODELS
        if model_mode == 1:
            with tf.variable_scope("Mono_1", reuse=tf.AUTO_REUSE):
                # model output
                self.num_unit = 128
                self.timestep = 20
                self.inpD_ = tf.argmax(self.inpD, axis=-1)
                self.inpM_ = tf.argmax(self.inpM, axis=-1)
                # self.yD = dur_model(self.inpD,
                #                     dur_dim=self.dur_dim,
                #                     emb_dim=self.emb_dim, 
                #                     num_unit=self.num_unit, 
                #                     scope="dur_model")
                self.yM = mnn_model(self.inpM,
                                    mnn_dim=self.mnn_dim,
                                    emb_dim=self.emb_dim,
                                    num_unit=self.num_unit, 
                                    scope="mnn_model")
                # loss
                self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
                self.learning_rate = 1e-3*1
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # dur model
                # self.lossD = tf.reduce_mean(-tf.reduce_sum(self.trueD * tf.log(self.yD+1e-8), axis=-1), name="dur_loss")
                # # self.update_opsD = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # # with tf.control_dependencies(self.update_opsD):
                # #     self.learning_rate = tf.train.exponential_decay(1e-3*1, self.global_step, 5000, 0.8, staircase=True)
                # self.train_opsD = self.optimizer.minimize(self.lossD, global_step=self.global_step)
                # mnn model
                self.lossM = tf.reduce_mean(-tf.reduce_sum(self.trueM * tf.log(self.yM+1e-8), axis=-1), name="mnn_loss")
                # # self.update_opsM = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # # with tf.control_dependencies(self.update_opsM):
                #     # self.learning_rate = tf.train.exponential_decay(1e-3*1, self.global_step, 5000, 0.8, staircase=True)
                self.train_opsM = self.optimizer.minimize(self.lossM, global_step=self.global_step)
                self.variables = tf.trainable_variables()
                self.gradients = self.optimizer.compute_gradients(self.lossM, self.variables)
                # metrics
                with tf.variable_scope("metrics/"):
                    # dur model
                    # self.predD_ = tf.argmax(self.yD, axis=-1)
                    # self.trueD_ = tf.argmax(self.trueD, axis=-1)
                    # self.correctD = tf.equal(self.predD_, self.trueD_)
                    # self.accD = tf.reduce_mean(tf.cast(self.correctD, tf.float32), name='dur_acc')
                    # mnn model
                    self.predM_ = tf.argmax(self.yM, axis=-1)
                    self.trueM_ = tf.argmax(self.trueM, axis=-1)          
                    self.correctM = tf.equal(self.predM_, self.trueM_)
                    self.accM = tf.reduce_mean(tf.cast(self.correctM, tf.float32), name='mnn_acc')

        if model_mode == 2:
            with tf.variable_scope("Mono_2", reuse=tf.AUTO_REUSE):
                # model output
                self.num_unit = 128
                self.timestep = 48
                self.inpM_o = tf.one_hot(self.inpM, self.inp_dim)
                self.trueM_o = tf.one_hot(self.trueM, self.inp_dim)
                self.yM = simple_model(self.inpM_o,
                                       inp_dim=self.inp_dim,
                                       emb_dim=self.emb_dim,
                                       num_unit=self.num_unit, 
                                       scope="simple_model")
                # loss
                self.lossM = tf.reduce_mean(-tf.reduce_sum(self.trueM_o * tf.log(self.yM+1e-8), axis=-1), name="loss")
                self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
                # self.update_opsM = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(self.update_opsM):
                    # self.learning_rate = tf.train.exponential_decay(1e-3*1, self.global_step, 5000, 0.8, staircase=True)
                self.learning_rate = 1e-3*1
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.variables = tf.trainable_variables()
                self.gradients = self.optimizer.compute_gradients(self.lossM, self.variables)
                self.train_opsM = self.optimizer.minimize(self.lossM, global_step=self.global_step)
                # metrics
                with tf.variable_scope("metrics/"):
                    self.predM_ = tf.argmax(self.yM, axis=-1)
                    self.trueM_ = tf.argmax(self.trueM_o, axis=-1)
                    self.correctM = tf.equal(self.predM_, self.trueM_)
                    self.accM = tf.reduce_mean(tf.cast(self.correctM, tf.float32), name='acc')

def f1_score(pred, true, mean=True):
    precision = list()
    recall = list()
    f1_score = list()
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_label = np.where(np.round(pred[i][j]) == 1.)[0]
            true_label = np.where(true[i][j] == 1.)[0]
            if len(pred_label) > 0:
                p_ = np.sum([len(np.where(pred_label == t)[0]) for t in true_label])
                r_ = np.sum([len(np.where(true_label == p)[0]) for p in pred_label])
                f1_ = 2 * (p_ * r_) / (p_ + r_ + 1e-8)
            else: p_, r_, f1_ = 0., 0., 0.
            precision.append(p_)
            recall.append(r_)
            f1_score.append(f1_)
    return precision, recall, f1_score


# MAIN EXPERIMENT
def main():
    # ATTRIBUTES
    batch_size, num_epoch = 8, 50
    model_mode = 1
    data_mode = "note"
    batch_mode = None

    # LOAD DATA
    if data_mode == "note":
        datapath = '/home/rsy/Documents/MIREX/dataset/PPDD_parsed_mono_small/prime/note/tfrecords'
    if data_mode == "frame":
        datapath = '/home/rsy/Documents/MIREX/dataset/PPDD_parsed_mono_small/prime/' + \
        'frame/%s/tfrecords' % (batch_mode)
    filenames_train = sorted(glob(os.path.join(datapath, 'train', '*.tfrecords')))
    filenames_val = sorted(glob(os.path.join(datapath, 'val', '*.tfrecords')))
    # get record length and steps
    train_len, val_len = 0, 0
    for d in filenames_train:
        for _ in tf.python_io.tf_record_iterator(d):
            train_len += 1
    for d in filenames_val:
        for _ in tf.python_io.tf_record_iterator(d):
            val_len += 1
    train_steps = int(np.ceil(train_len / batch_size))
    val_steps = int(np.ceil(val_len / batch_size))
    # data queue with tfrecord
    parse_tf = ParseTfrecords(train_path=filenames_train,
                              val_path=filenames_val,
                              num_epoch=num_epoch,
                              batch_size1=batch_size,
                              batch_size2=val_len,
                              data_mode=data_mode,
                              batch_mode=batch_mode)
    train_data, val_data = parse_tf()
    print('PARSED TFRECORDS')
    # load random data
    # rand_inp = np.load("rand_inp_88.npy")[:,:48]
    # rand_oup = np.load("rand_oup_88.npy")[:,:48]
    # rand_inp_val = np.load("rand_inp_88_val.npy")[:,:48]
    # rand_oup_val = np.load("rand_oup_88_val.npy")[:,:48]
    # train_data = None

    # LOAD GRAPH
    g = Graph(train_data, data_mode=data_mode, batch_mode=batch_mode, model_mode=model_mode)
    saver = tf.train.Saver()
    train_loss_all = list()
    train_loss_step = list()
    val_loss_step = list()
    gradient_step = list()
    min_val_loss = 100
    early_stopping = 0
    i = 0
    print('LOADED GRAPH')

    # with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('START TRAINING...')
        try:
            while True:
                if model_mode == 1:
                    is_training = g.phase
                    feed_dict = {is_training: True}
                    # _, gradients, predD, inpD, trueD, train_lossD, train_accD, step = sess.run(
                    #     [g.train_opsD, g.gradients, g.yD, g.inpD_, g.trueD_, g.lossD, g.accD, g.global_step], 
                    #     feed_dict=feed_dict)
                    # train_loss_all.append([train_lossD, train_accD])
                    # gradient_step.append([np.mean(gradients[i][0]) for i in range(0,len(gradients),2)])
                    _, gradients, predM, inpM, trueM, train_lossM, train_accM, step = sess.run(
                        [g.train_opsM, g.gradients, g.yM, g.inpM_, g.trueM_, g.lossM, g.accM, g.global_step], 
                        feed_dict=feed_dict)
                    train_loss_all.append([train_lossM, train_accM])
                    gradient_step.append([np.mean(gradients[i][0]) for i in range(0,len(gradients),2)])
                    # save results
                    if step % 10 == 0:
                        print('step: %d / loss: %f / acc: %f' \
                        # % (step, train_lossD, train_accD))
                        % (step, train_lossM, train_accM))
                        # print('step: %d / loss(dur): %f / loss(mnn): %f / acc(dur): %f / acc(mnn): %f' \
                        # % (step, train_lossD, train_lossM, train_accD, train_accM))
                        # print("input:")
                        # print(inpD[0])
                        # print("ground truth")
                        # print(trueD[0])
                        # print("output:")
                        # print(np.argmax(predD[0], axis=-1))
                        print("input:")
                        print(inpM[0])
                        print("ground truth")
                        print(trueM[0])
                        print("output:")
                        print(np.argmax(predM[0], axis=-1))
                        print("gradients:")
                        print(gradients[0])
                    if step % 1000 == 0:
                        # train_loss_step.append([train_lossD, train_accD])
                        train_loss_step.append([train_lossM, train_accM])
                        # validation
                        # vdi, vdo, _, _ = sess.run(val_data)
                        # d_inp, d_oup, is_training = g.inpD, g.trueD, g.phase
                        # feed_dict = {d_inp: vdi, d_oup: vdo, is_training: True}
                        # val_predD, val_lossD, val_accD = sess.run(
                        #     [g.yD, g.lossD, g.accD], feed_dict=feed_dict)
                        # val_loss_step.append([val_lossD, val_accD])
                        _, _, vmi, vmo = sess.run(val_data)
                        m_inp, m_oup, is_training = g.inpM, g.trueM, g.phase 
                        feed_dict = {m_inp: vmi, m_oup: vmo, is_training: True}
                        val_predM, val_lossM, val_accM = sess.run(
                            [g.yM, g.lossM, g.accM], feed_dict=feed_dict)
                        val_loss_step.append([val_lossM, val_accM])
                        print('Val: step: %d / loss: %f / acc: %f' \
                        # % (step, val_lossD, val_accD))
                        % (step, val_lossM, val_accM))
                        # print('Val: step: %d / loss(dur): %f / loss(mnn): %f / acc(dur): %f / acc(mnn): %f' \
                        # % (step, val_lossD, val_lossM, val_accD, val_accM))
                        # if val_lossD < min_val_loss:
                        #     saver.save(sess, "./Monophonic_model_1_ckpt_1", global_step=step)
                        #     min_val_loss = val_lossD
                        #     early_stopping = 0
                        # else: early_stopping += 1
                        # save losses
                        np.save("mono_model_1_mnn_train_loss_3.npy", train_loss_step)
                        np.save("mono_model_1_mnn_val_loss_3.npy", val_loss_step)
                        np.save("mono_model_1_mnn_loss_every_step_3.npy", train_loss_all)
                        np.save("mono_model_1_mnn_gradient_step_3.npy", gradient_step)
                        # # early stopping
                        # if early_stopping > 10:
                        #     break

                if model_mode == 2:
                    is_training = g.phase
                    feed_dict = {is_training: True}
                    # ti, to, is_training = g.inpM, g.trueM, g.phase
                    # inp = rand_inp[i+batch_size:i+2*batch_size]
                    # oup = rand_oup[i+batch_size:i+2*batch_size]
                    # feed_dict = {ti: inp, to: oup, is_training: True}
                    _, inpM, predM, trueM, gradients, train_lossM, train_accM, step = sess.run(
                        [g.train_opsM, g.inpM, g.yM, g.trueM, g.gradients, g.lossM, g.accM, g.global_step], 
                        feed_dict=feed_dict)
                    train_loss_all.append([train_lossM, train_accM])
                    gradient_step.append([np.mean(gradients[i][0]) for i in range(0,len(gradients),2)])
                    # save results
                    if step % 10 == 0:
                        print('step: %d / loss: %f / acc: %f' \
                        % (step, train_lossM, train_accM))
                        print("input:")
                        print(inpM[0])
                        print("ground truth")
                        print(trueM[0])
                        print("output:")
                        print(np.argmax(predM[0], axis=-1))
                        print("gradients:")
                        print(gradients[0])
                        # print("L0:")
                        # print(l0[0])
                        # print("L1:")
                        # print(l1[0])
                        # print("L2:")
                        # print(l2[0])
                        # print("L3:")
                        # print(l3[0])
                        if np.isnan(train_lossM) == True:
                            break
                    if step % 1000 == 0:
                        train_loss_step.append([train_lossM, train_accM])
                        # validation
                        vmi, vmo = sess.run(val_data)
                        # vmi = rand_inp_val
                        # vmo = rand_oup_val
                        m_inp, m_oup, is_training = g.inpM, g.trueM, g.phase
                        feed_dict = {m_inp: vmi, m_oup: vmo, is_training: True}
                        val_inpM, val_predM, val_lossM, val_accM = sess.run(
                            [g.inpM, g.yM, g.lossM, g.accM], feed_dict=feed_dict)
                        val_loss_step.append([val_lossM, val_accM])
                        print('Val: step: %d / loss: %f / acc: %f' \
                        % (step, val_lossM, val_accM))
                        print('Val input:')
                        print(val_inpM[0])
                        print('Val output:')
                        print(val_predM)
                        # if val_lossM < min_val_loss:
                        #     saver.save(sess, "./Monophonic_model_2_ckpt_1", global_step=step)
                        #     min_val_loss = val_lossM
                        #     early_stopping = 0
                        # else: early_stopping += 1
                        # save losses
                        np.save("mono_model_2_train_loss_1_96.npy", train_loss_step)
                        np.save("mono_model_2_val_loss_1_96.npy", val_loss_step)
                        np.save("mono_model_2_loss_every_step_1_96.npy", train_loss_all)
                        np.save("mono_model_2_gradient_step_1_96.npy", gradient_step)
                        # # early stopping
                        # if early_stopping > 10:
                        #     break

        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)

        finally:
            coord.request_stop()
            coord.join(threads)


def generation():
    timesteps = 96
    input_dim = 89 # pitch(88) + rest(1)
    data_mode = "frame"
    batch_mode = "reg"
    model_mode = 3
    datapath = '/home/rsy/Documents/MIREX/dataset/PPDD_mono_small/%s/test_%s' % (data_mode, batch_mode)
    ti1 = np.load(sorted(glob(os.path.join(datapath, "*_inp.npy")))[0])[8]
    ti2 = np.load(sorted(glob(os.path.join(datapath, "*_cond.npy")))[0])[8]
    to = np.load(sorted(glob(os.path.join(datapath, "*_oup.npy")))[0])[8]

    go_frame = np.reshape(ti1[-1], (1, 1))
    enc_seed = np.expand_dims(ti1, 0)
    dec_seed = np.expand_dims(ti2, 0)
    generated = list()
    gen_step = 96
    graph = tf.get_default_graph()
    last_pred = 99

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.import_meta_graph('./Monophonic_model_1_ckpt_1.meta')
        saver.restore(sess, './Monophonic_model_1_ckpt_1-14000')
        writer = tf.summary.FileWriter('./mono_graph_1')
        writer.add_graph(sess.graph)
        inp1 = graph.get_tensor_by_name("enc_in:0")
        inp2 = graph.get_tensor_by_name("cond_in:0")
        phase = graph.get_tensor_by_name("phase:0")
        oup = graph.get_tensor_by_name("Mono_1/dur_model/final_out:0")
        # att = graph.get_tensor_by_name("Monophonic_3/model/output_decoder/attention/sigmoid:0")
        # prediction
        for i in range(gen_step):
            feed_dict = {inp1: enc_seed, inp2: dec_seed, phase: False}
            preds = sess.run(oup, feed_dict=feed_dict)
            sampled_pred = np.argmax(preds[0][-1])
            if last_pred == sampled_pred:
                onset = 0
            else: onset = 1
            enc_seed = np.concatenate([enc_seed, np.reshape(sampled_pred, (1,1))], axis=1)
            dec_seed = np.concatenate([dec_seed, np.reshape(onset, (1,1))], axis=1)
            enc_seed = enc_seed[:,1:]
            dec_seed = dec_seed[:,1:]
            last_pred = sampled_pred
            print("generated %dth sample" % (i+1), end='\r')
        # beam search
        k = 3
        seq_scores = [[list(), 1.0]]
        for row in preds[0]: # each timestep
            all_candidates = list()
            for j in range(len(seq_scores)):
                seq, score = seq_scores[j]
                for l in range(len(row)): # consider each class
                    candidate = [seq + [l], score * -np.log(row[l]+1e-8)]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda up:up[1]) # sort by score
            seq_scores = ordered[:k]
    return generated, att_mtx, seq_scores


def sample(preds, temperature=None): # multinomial sampling for generation
    if temperature is not None:
        log_preds = np.log(preds + 1e-7) / temperature
        exp_preds = np.exp(log_preds)
        normalize_preds = exp_preds / np.sum(np.abs(exp_preds) + 1e-7)
        probas = np.random.multinomial(1, normalize_preds)
    else: probas = preds
    '''
    Ref:
    "Deep voice: real-time neural TTS (Arik et al., 2017)"
    "Deep artificial composer: a creative neural network model for automated meloty generation (Colombo et al., 2017)"
    '''
    return np.argmax(probas)



# self.num_pos = tf.expand_dims(tf.reduce_sum(self.yM, axis=-1), -1)
# self.num_neg = self.input_dim - self.num_pos
# self.pos_w = self.num_neg / self.num_pos
# self.weignted_binary_ce = tf.add((1 - self.trueM) * self.yM,
#  (1 + (self.pos_w - 1) * self.trueM) * (tf.log(1 + tf.exp(-tf.abs(self.yM))) + tf.nn.relu(-self.yM)),
#  name="weighted_binary_ce_loss")
# self.binary_ce = -tf.reduce_sum(
#     (self.trueM * tf.log(self.yM+1e-8)) + ((1-self.trueM) * tf.log(1-self.yM+1e-8)), axis=-1,
#     name="binary_ce_loss")
# self.l1_loss = tf.abs(self.trueM - self.yM)
# self.pred = tf.round(self.yM, name='prediction')
# self.num_correct = tf.equal(self.pred, self.trueM)
# self.acc = tf.cast(self.num_correct, tf.float32)
# self.loss = tf.reduce_mean(1 - self.acc, name="model_loss") # hamming loss



if __name__ == "__main__":
    main()
