import os
import numpy as np
import random
random.seed(1234)

"""
Very old style but easy to implement and verify a quick experiment.
"""


class Vad_Data_Provider():
    def __init__(self,
                 data_dir,
                 set_name=None,
                 label_ext = ".label.npy",
                 feats_ext = ".mfcc_vadtype1.npy",
                 seq_len=500,
                 return_sequences=False,
                 batch_size=512):
        """Data Provider for VAD data.
        Very old style but easy to implement and verify a quick experiment.
        TODO: to support RNN

        :param data_dir:  this could be either the training set dir,
                          the valid set dir, or the test set dir.
        :param set_name: 
        :param label_ext: 
        :param feats_ext: 
        :returns: 
        :rtype: 

        """
        self.data_dir = data_dir
        self.label_ext = label_ext
        self.feats_ext = feats_ext
        self.return_sequences = return_sequences
        self.seq_len = seq_len
        self.batch_size = batch_size

        # use all data or a specific category?
        file_list = os.listdir(self.data_dir)
        self.feats_fn_list = [fn for fn in file_list if fn.endswith(self.feats_ext)]
        self.label_fn_list = [fn for fn in file_list if fn.endswith(self.label_ext)]
        if not set_name:
            pass
        else:
            self.feats_fn_list = [fn for fn in file_list if set_name in fn]
            self.label_fn_list = [fn for fn in file_list if set_name in fn]
        print("----Start init Data----")
        print("feats_fn_list: %s" % str(feats_fn_list))
        print("label_fn_list: %s" % str(label_fn_list))

        # Load data and to rnn or dnn.
        # rnn format: [?, fix_seq_len, dim]
        # dnn format: [?, dim]
        self.feats = []
        self.labels = []
        for fn_feats, fn_label in zip(feats_fn_list, label_fn_list):
            feats_data = np.load(os.path.join(data_dir, fn_feats))
            label_data = np.load(os.path.join(data_dir, fn_label))
            if self.return_sequences:
                feat_data, label_data = self._feats_label_to_seq(feats_data, label_data)
            self.feats.append(feat)
            self.labels.append(label)
        self.feats = np.concatenate(self.feats)
        self.labels = np.concatenate(self.labels)

        # Index.
        self.total_sample_nb = len(self.labels)
        self.idx = 0
        self.idxes = list(np.arange(self.total_sample_nb))
        self.pointer = 0
        self.epoch = 0

    def _feats_label_to_seq(self, feats_data, label_data):
        """[n, m] dnn style converted to RNN style.

        :param feats_data: frame level [n, m_feats]
        :param label_data: frame level [n, m_label]
        :returns: 
        :rtype: 

        """
        offset = len(feats_data) % self.seq_len
        if offset != 0:
            feats_data_rnn = np.zeros((len(feats_data) + offset, feats_data.shape[-1]))
            feats_data_rnn[:len(feats_data), :] = feats_data
            label_data_rnn = np.zeros((len(feats_data) + offset, feats_data.shape[-1]))
            label_data_rnn[:len(label_data), :] = label_data
        feats_data_rnn = feats_data_rnn.reshape((-1, self.seq_len, feats_data.shape[-1]))
        label_data_rnn = label_data_rnn.reshape((-1, self.seq_len, label_data.shape[-1]))

        return feats_data_rnn, label_data_rnn

    def get_batch(self):
        # The last batch.
        if self.idx+self.batch_size > self.total_sample_nb:
            self._shuffle_data()
            feat = self.feats[-self.batch_size:]
            label = self.labels[-self.batch_size:]
            self.idx = 0
            self.epoch += 1
        # Normal batch iterations.
        else:
            feat = self.feats[self.idxes[idx: idx+self.batch_size]]
            label = self.labels[self.idxes[idx: idx+self.batch_size]]
            self.idx += self.batch_size

        return feat, label

    def _shuffle_data(self):
        random.shuffle(self.idxes)

