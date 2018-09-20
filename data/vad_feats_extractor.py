import librosa
import scipy.io
import numpy as np
import argparse
import os
import soundfile as sf

"""
INPUT:
VAD type-1 feats:
MFCC
delta
delta delta
energy

OUTPUT:
labesl: are basically [1, 0, 0, 1, ....]
"""

class Feature_Extractor(object):

    def __init__(self,
                 input_dir,
                 output_dir,
                 valid_percent,
                 test_percent):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.valid_percent = valid_percent
        self.test_percent = test_percent

        if not os.path.exists(self.input_dir):
            raise ValueError("Invalid input data dir %s" % self.input_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(args.out_data_dir)
        if (valid_percent + test_percent) >= 1:
            raise ValueError("Inavlid valid_percent %f and INvalid test_percent %f" % (
                valid_percent, test_percent))

    def extract_vad_type1_features(self,
                                   label_ext=".mat",
                                   n_mfcc = 12,
                                   sampling_rate=16000,
                                   frame_size=256,
                                   frame_step=128):
        """ 1) Extract MFCC features from VAD dataset in https://github.com/jtkim-kaist/VAD
        2) Do alignment if necessary.

        vad_type1: MFCC + delta + energy

        Typical setting:
        -sample input files at 16kHz
        -number of MFCC: 12
        -frame szie: 256
        -hop size(frame_step): 128
        """
        print("-----Extrats VAD type1 (mfcc-based) features-----")
        if label_ext != ".mat":
            raise ValueError("label_ext %s does not support" % label_ext)
        if not os.path.exists(os.path.join(self.output_dir, "train")):
            os.mkdirs(os.path.join(self.output_dir, "train"))
        if not os.path.exists(os.path.join(self.output_dir, "valid")):
            os.mkdirs(os.path.join(self.output_dir, "valid"))
        if not os.path.exists(os.path.join(self.output_dir, "test")):
            os.mkdirs(os.path.join(self.output_dir, "test"))

        wav_list = os.listdir(self.input_dir)
        wav_list = [fn for fn in wav_list if fn.endswith(".wav")]
        label_list = [name.split(".")[0]+label_ext for name in wav_list]

        for wav_nm, label_nm in zip(wav_list, label_list):
            label_path = os.path.join(self.input_dir, label_nm)
            wav_path = os.path.join(self.input_dir, wav_nm)
            # Load the audio time series and its sampling rate, labels.

            # =================
            #    feats analysis
            # =================
            sound_clip, s = sf.read(wav_path)
            print("loadded wav shape: %s" % str(sound_clip.shape))
            # Mel Frequency Cepstral Coefficents
            mfcc = librosa.feature.mfcc(
                y=sound_clip,
                sr=sampling_rate,
                n_mfcc=n_mfcc,
                n_fft=frame_size,
                hop_length=frame_step)

            # MFCC and deltas
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            # Energy
            mel_spectogram = librosa.feature.melspectrogram(
                y=sound_clip,
                sr=sampling_rate,
                n_fft=frame_size,
                hop_length=frame_step)
            rmse = librosa.feature.rmse(
                S=mel_spectogram,
                frame_length=frame_size,
                hop_length=frame_step)

            mfcc = np.asarray(mfcc)
            mfcc_delta = np.asarray(mfcc_delta)
            mfcc_delta2 = np.asarray(mfcc_delta2)
            rmse = np.asarray(rmse)

            feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)
            feature = feature.T

            print(feature.shape)

            # ======================
            # sample to frame label
            # ======================
            labels_sample = scipy.io.loadmat(label_path)['y_label'].reshape((-1))
            print("loadded label shape:%s" % str(labels_sample.shape))
            len_labels_sample = len(labels_sample)
            # Padding to make it evenly divided by frame step.
            if len_labels_sample % frame_step != 0:
                labels_sample_placeholder = np.zeros((len_labels_sample // frame_step) * frame_step + frame_step)
                labels_sample_placeholder[:len_labels_sample] = labels_sample
            # sample 2 frame.
            labels_sample = labels_sample_placeholder.reshape((-1, frame_step))
            labels_sample = np.sum(labels_sample, axis=-1)
            labels_frm = np.array(labels_sample >= (frame_step // 2), dtype=np.float32)
            labels_frm = labels_frm.reshape((-1, 1))

            # ===============
            # alignment
            # ==============
            common_len = np.min([len(labels_frm), len(feature)])
            labels_frm = labels_frm[:common_len, :]
            feature = feature[:common_len, :]
            print("label final shape: %s", str(labels_frm.shape))
            print("feature final shape: %s", str(feature.shape))

            # ====================
            #      Dump
            # ===================
            basename = wav_nm.split(".")[0]
            # Split to train, vad, test dirs and do dump.
            len_data = len(feature)
            feature_valid = feature[:self.valid_percent * len_data, :]
            feature_test = feature[-self.test_percent: * len_data, :]
            feature_train = feature[self.valid_percent:-self.test_percent:, :]
            labels_frm_valid = labels_frm[:self.valid_percent * len_data, :]
            labels_frm_test = labels_frm[-self.test_percent: * len_data, :]
            labels_frm_train = labels_frm[self.valid_percent:-self.test_percent:, :]
            np.save(os.path.join(self.output_dir, "train", basename+".mfcc_vadtype1.npy"), feature_train)
            np.save(os.path.join(self.output_dir, "train", basename+".label.npy"), labels_frm_train)
            np.save(os.path.join(self.output_dir, "valid", basename+".mfcc_vadtype1.npy"), feature_valid)
            np.save(os.path.join(self.output_dir, "valid", basename+".label.npy"), labels_frm_valid)
            np.save(os.path.join(self.output_dir, "test", basename+".mfcc_vadtype1.npy"), feature_test)
            np.save(os.path.join(self.output_dir, "test", basename+".label.npy"), labels_frm_test)
            print("-------%s Processed Done.------" % basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Feats extractor for for data set: https://github.com/jtkim-kaist/VAD")
    parser.add_argument('--in_raw_dir',
                        help='input raw data dir, containing wav. and mat (basically matlab format to hold the labels.')
    parser.add_argument('--out_data_dir',
                        help='aligned inoput and output feats in numpy format with same basename.')
    args = parser.parse_args()
   
    feat_ext = Feature_Extractor(input_dir=args.in_raw_dir, output_dir=args.out_data_dir)
    feat_ext.extract_vad_type1_features()
