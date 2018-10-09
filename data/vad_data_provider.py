import os
import numpy as np
import random
import argparse
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

        :param data_dir:  this could be either the training set di,
                          the valid set dir, or the test set dir.
        :param set_name: 
        :param label_ext: 
        :param feats_ext: 
        :returns: 
        :rtype: 

        """
        self.output_dim = 2
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
        print("----Start init Data (%s)----" % self.data_dir)
        print("feats_fn_list: %s" % str(self.feats_fn_list))
        print("label_fn_list: %s" % str(self.label_fn_list))

        # Load data and to rnn or dnn.
        # rnn format: [?, fix_seq_len, dim]
        # dnn format: [?, dim]
        self.feats = []
        self.labels = []
        for fn_feats, fn_label in zip(self.feats_fn_list, self.label_fn_list):
            feats_data = np.load(os.path.join(data_dir, fn_feats))
            label_data = np.load(os.path.join(data_dir, fn_label))
            label_data = np.array(label_data, dtype=np.int32)
            if self.return_sequences:
                feat_data, label_data = self._feats_label_to_seq(feats_data, label_data)
            self.feats.append(feats_data)
            self.labels.append(label_data)
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
            feat = self.feats[-self.batch_size:]
            label = self.labels[-self.batch_size:]
            self._shuffle_data()
            self.idx = 0
            self.epoch += 1
        # Normal batch iterations.
        else:
            feat = self.feats[self.idxes[self.idx: self.idx+self.batch_size]]
            label = self.labels[self.idxes[self.idx: self.idx+self.batch_size]]
            self.idx += self.batch_size

        return feat, label

    def _shuffle_data(self):
        random.shuffle(self.idxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data provider for for data set: https://github.com/jtkim-kaist/VAD")
    parser.add_argument('--train_dir', type=str, default="/home/kingstorm/dataset/vad_3rd_train_test/train",
                        help='dir holding training set.')
    parser.add_argument('--valid_dir', type=str, default="/home/kingstorm/dataset/vad_3rd_train_test/valid",
                        help='dir holding validation set.')
    parser.add_argument('--test_dir', type=str, default="/home/kingstorm/dataset/vad_3rd_train_test/test",
                        help='dir holding test set.')
    args = parser.parse_args()

    vad_train_provider = Vad_Data_Provider(data_dir=args.train_dir)
    vad_valid_provider = Vad_Data_Provider(data_dir=args.valid_dir)
    vad_test_provider = Vad_Data_Provider(data_dir=args.test_dir)

    import pdb
    pdb.set_trace()
    iterations = 10000
    for itr in np.arange(iterations):
        temp_train = vad_train_provider.get_batch()
        temp_valid = vad_valid_provider.get_batch()
        temp_test = vad_test_provider.get_batch()

