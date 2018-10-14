from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python import debug as tf_debug

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.binary_net import BinaryNet
from utils import handle_args
from data.vad_data_provider import Vad_Data_Provider

BN_TRAIN_PHASE = True
BN_TEST_PHASE = False


"""
THis is just a DEMO script.
"""

# =======================
#      hyper parameter
# =======================
INPUT_DIM = 37
OUTPUT_DIM = 2


# ======================
#       Plot utils
# ======================
def plt2image(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)

def plot_vad(label, label_, show=5000):
    '''
    label: [0, 1, 0, 0, 1...]
    label: [0, 1, 1, 1, 1...]
    '''
    fig = plt.figure()
    fig.clf()
    plt.plot(label[:show]+0.2, label="label")
    plt.plot(label_[:show], label="label_hat")
    plt.legend()
    formattedImage = plt2image(fig)
    plt.close(fig)
    return formattedImage

def add_vad_plot_summary(sess, writer, placeholder, image_summ, label, label_, step):
    """
    label:  [batch, ?, output_dim]
    label_: [batch, ?, output_dim]
    """
    label = label.reshape((-1, OUTPUT_DIM))
    label_ = label_.reshape((-1, OUTPUT_DIM))
    label = np.array((label > 0.5)[:, 1], dtype=np.float32)
    label_ = np.array((label_ > 0.5)[:, 1], dtype=np.float32)
    vad_image = plot_vad(label, label_)

    vad_plot_summ, _ = sess.run([image_summ, placeholder],
                                feed_dict={placeholder: vad_image})
    writer.add_summary(vad_plot_summ, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="", help='directory for storing input data')
    parser.add_argument(
        '--log_dir', help='root path for logging events and checkpointing')
    parser.add_argument(
        '--extra', help='for specifying extra details (e.g one-off experiments)')
    parser.add_argument(
        '--n_hidden', help='number of hidden units', type=int, default=512)
    parser.add_argument(
        '--keep_prob', help='dropout keep_prob', type=float, default=0.8)
    parser.add_argument(
        '--reg', help='how much to push weights to +1/-1', type=float, default=0.5)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=1e-5)
    parser.add_argument(
        '--batch_size', help='examples per mini-batch', type=int, default=128)
    parser.add_argument(
        '--max_steps', help='maximum training steps', type=int, default=1000000)
    parser.add_argument(
        '--gpu', help='physical id of GPUs to use')
    parser.add_argument(
        '--eval_every_n', help='validate model every n steps', type=int, default=1000)
    parser.add_argument(
        '--binary', help="should weights and activations be constrained to -1, +1", action="store_true")
    parser.add_argument(
        '--first', help="also binarize first layer (requires --binary)", action="store_true")
    parser.add_argument(
        '--last', help="also binarize last layer (requires --binary)", action="store_true")
    parser.add_argument(
        '--xnor', help="if binary flag is passed, determines if xnor_gemm cuda kernel is used to accelerate training, otherwise no effect", action="store_true")
    parser.add_argument(
        '--batch_norm', help="batch normalize activations", action="store_true")
    parser.add_argument(
        '--debug', help="run with tfdbg", action="store_true")
    parser.add_argument(
        '--restore', help='where to load model checkpoints from')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_path, binary, first, last, xnor, batch_norm = handle_args(args)

    # import data
    #mnist = input_data.read_data_sets(
    #    args.data_dir, dtype=tf.float32, one_hot=True)
    vad_train_provider = Vad_Data_Provider(data_dir="/home/kingstorm/dataset/vad_3rd_train_test/train", set_name="room")
    # TODO: There is a problem with valid/test sets, now just use test set
    vad_valid_provider = Vad_Data_Provider(data_dir="/home/kingstorm/dataset/vad_3rd_train_test/test", set_name="room")
    vad_test_provider = Vad_Data_Provider(data_dir="/home/kingstorm/dataset/vad_3rd_train_test/test", set_name="room")
    dtype = tf.float32

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        x = tf.placeholder(dtype, [None, INPUT_DIM])
        phase = tf.placeholder(tf.bool, name='phase')
        keep_prob = tf.placeholder(tf.float32)

        # create the model
        bnn = BinaryNet(binary, first, last, xnor, args.n_hidden,
                        keep_prob, x, batch_norm, phase, OUTPUT_DIM, INPUT_DIM)
        y = bnn.output
        y_placeholder = tf.placeholder(tf.int32, [None, 1])
        y_ = tf.one_hot(y_placeholder, OUTPUT_DIM)
        y_ = tf.reshape(y_, [-1, OUTPUT_DIM])

        # define loss and optimizer
        if binary:
            weight_penalty = bnn.W_2_p + bnn.W_3_p
            if first:
                weight_penalty += bnn.W_1_p
            if last:
                weight_penalty += bnn.W_out_p
            total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) + args.reg * weight_penalty
        else:
            total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # for batch-normalization
        if batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # ensures that we execute the update_ops before performing the
                # train_op
                train_op = tf.contrib.layers.optimize_loss(
                    total_loss, global_step, learning_rate=args.lr, optimizer='Adam',
                    summaries=["gradients"])
        else:
            train_op = tf.contrib.layers.optimize_loss(
                total_loss, global_step, learning_rate=args.lr, optimizer='Adam',
                summaries=["gradients"])

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(tf.global_variables())

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if args.debug:
            print("Using debug mode")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

       # setup summary writer
        if args.log_dir:
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)
            training_summary = tf.summary.scalar("train loss", total_loss)
            test_summary = tf.summary.scalar("test acc.", accuracy)
            merge_op = tf.summary.merge_all()
            # Plot summary.
            vad_vis = tf.placeholder(tf.uint8)
            vad_vis_summary = tf.summary.image("vad plot", vad_vis)

        if args.restore:
            saver.restore(sess, tf.train.latest_checkpoint(args.restore))
            # os.path.join(log_path, args.restore)))
            init_step = sess.run(global_step)
            print('Restoring network previously trained to step %d' % init_step)
        else:
            init_step = 0

        # Train
        timing_arr = np.zeros(args.max_steps)
        step = init_step
        while step < init_step + args.max_steps:

            batch_xs, batch_ys = vad_train_provider.get_batch()

            start_time = time.time()
            __, loss = sess.run([train_op, total_loss], feed_dict={
                x: batch_xs, y_placeholder: batch_ys, keep_prob: args.keep_prob, phase: BN_TRAIN_PHASE})
            timing_arr[step - init_step] = time.time() - start_time

            if step % args.eval_every_n == 0:
                if xnor:
                    test_batch_xs, test_batch_ys = vad_valid_provider.feats, vad_valid_provider.labels
                    if args.log_dir:
                        test_acc, merged_summ, y_hat_v, y_v = sess.run(
                            [accuracy, merge_op, y, y_], feed_dict={
                                x: test_batch_xs,
                                y_placeholder: test_batch_ys,
                                keep_prob: 1.0, phase: BN_TEST_PHASE})
                        add_vad_plot_summary(sess, summary_writer, vad_vis, vad_vis_summary,
                                             y_v, y_hat_v, step)
                    else:
                        test_acc = sess.run(accuracy, feed_dict={
                            x: test_batch_xs, y_placeholder: test_batch_ys, phase: BN_TEST_PHASE})
                else:
                    if args.log_dir:
                        test_acc, merged_summ, y_hat_v, y_v = sess.run(
                            [accuracy, merge_op, y, y_], feed_dict={
                                x: vad_valid_provider.feats,
                                y_placeholder: vad_valid_provider.labels,
                                keep_prob: 1.0, phase: BN_TEST_PHASE})
                        add_vad_plot_summary(sess, summary_writer, vad_vis, vad_vis_summary,
                                             y_v, y_hat_v, step)

                    else:
                        test_acc = sess.run(accuracy, feed_dict={
                            x: vad_valid_provider.feats,
                            y_placeholder: vad_valid_provider.labels,
                            keep_prob: 1.0, phase: BN_TEST_PHASE})
                print("epoch %d step %d, loss = %.4f, test accuracy %.4f (%.1f ex/s)" % (
                    vad_train_provider.epoch, step, loss, test_acc,
                    float(args.batch_size / timing_arr[step - init_step])))

                if args.log_dir:
                    summary_writer.add_summary(merged_summ, step)
                    summary_writer.flush()
            step += 1

        # Test trained model
        if not xnor:
            print("Final test accuracy %.4f" % (sess.run(accuracy, feed_dict={x: vad_test_provider.feats,
                                                                              y_placeholder: vad_test_provider.labels,
                                                                              keep_prob: 1.0,
                                                                              phase: BN_TEST_PHASE})))
        print("Avg ex/s = %.1f" % float(args.batch_size / np.mean(timing_arr)))
        print("Med ex/s = %.1f" %
              float(args.batch_size / np.median(timing_arr)))

        if args.log_dir:
            # save model checkpoint
            checkpoint_path = os.path.join(log_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            sess.close()
