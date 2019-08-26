import timeit
import numpy as np
from scipy.io import wavfile
import pickle
import random
import math
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

start = timeit.default_timer()

train_num = 40000
test_num = 40000

init_channel = 8

l2_loss_list = []

def read_filename():
  path = '/home/chenbolin/speech/{folder}/Speech_{n1}_{n2}.wav'

  cover_folder = 'speech_cover1s'
  stego_folder = 'speech_stego50'

  cover_audio_list = []

  for i in xrange(10000):
    for j in xrange(4):
      filepath = path.format(folder = cover_folder, n1 = i + 1, n2 = j + 1)
      cover_audio_list.append(filepath)

  random.shuffle(cover_audio_list)

  print('Shuffle filename complete')

  audio_name_list = []

  for i in xrange(len(cover_audio_list)):
    cover_name = cover_audio_list[i]
    stego_name = cover_name.replace(cover_folder, stego_folder)
    audio_name_list.append(cover_name)
    audio_name_list.append(stego_name)



  # print('Append filename complete')

  #select data
  # train_num = 40000
  # test_num = 40000

  train_audios_name = audio_name_list[: train_num]
  test_audios_name = audio_name_list[train_num : train_num + test_num]

  # print('Select data complete')

  return train_audios_name, test_audios_name



def read_data():
  train_audios_name, test_audios_name = read_filename()

  train_audios_data = []
  for i in xrange(len(train_audios_name)):
    audio_data = wavfile.read(train_audios_name[i])[1]

    train_audios_data.append(audio_data)

  test_audios_data = []
  for i in xrange(len(test_audios_name)):
    audio_data = wavfile.read(test_audios_name[i])[1]

    test_audios_data.append(audio_data)

  print('Read data complete')

  train_labels_data = [0, 1] * (train_num / 2)
  test_labels_data = [0, 1] * (test_num / 2)

  train_audios_data = np.asarray(train_audios_data)
  train_labels_data = np.asarray(train_labels_data)
  test_audios_data = np.asarray(test_audios_data)
  test_labels_data = np.asarray(test_labels_data)


  return train_audios_data, test_audios_data, train_labels_data, test_labels_data

def weight_variable(shape,):
  initial = tf.truncated_normal(shape, stddev=0.1)
  weight = tf.Variable(initial, name='weight')

  l2_loss_list.append(tf.nn.l2_loss(weight))

  return weight

def bias_variable(shape):
  initial = tf.constant(0.0, shape = shape)
  return tf.Variable(initial, name='bias')

def compute_time():
  stop = timeit.default_timer()
  seconds = stop - start
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  d, h = divmod(h, 24)
  print('Run time: %d:%02d:%02d:%02d' % (d, h, m, s))

def conv_act(input, w_shape, b_shape, name):
  with tf.variable_scope(name):
    w_conv = weight_variable(w_shape)
    b_conv = bias_variable(b_shape)

    z_conv = tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv
    h_conv = tf.nn.tanh(z_conv)

    return h_conv

def conv_without_act(input, w_shape, b_shape, name):
  with tf.variable_scope(name):
    w_conv = weight_variable(w_shape)
    b_conv = bias_variable(b_shape)

    z_conv = tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv

    return z_conv


def conv_subsample_without_act(input, channel, name):
  with tf.variable_scope(name):
    w_conv = weight_variable([1, 3, channel, channel])
    b_conv = bias_variable([channel])

    z_conv = tf.nn.conv2d(input, w_conv, strides=[1, 1, 2, 1], padding='SAME') + b_conv

    return z_conv


def inference(audios):
  audios_reshape = tf.reshape(audios, [-1, 1, 16000, 1])

  filter_kernel = np.array([-1, 2, -1])
  filter_kernel = filter_kernel.reshape((1, filter_kernel.shape[0], 1, 1))

  kernel_variable = tf.constant(filter_kernel, dtype='float', name='filter_kernel')

  # filter = tf.constant([-1, 2, -1], dtype='float', name='preprocess_filter')

  # filter_pack1 = tf.pack([filter], axis=0)
  # filter_pack2 = tf.pack([filter_pack1], axis=2)
  # filter_pack3 = tf.pack([filter_pack2], axis=3)

  audios_preprocess = tf.nn.conv2d(audios_reshape, filter_kernel, strides=[1, 1, 1, 1], padding='VALID')

  with tf.variable_scope('group1'):
    output = conv_without_act(audios_preprocess, [1, 5, 1, 1], [1], name='conv1')
    output = conv_without_act(output, [1, 1, 1, init_channel], [init_channel], name='conv2')
    output = conv_subsample_without_act(output, init_channel, name='subsample')


  with tf.variable_scope('group2'):
    output = conv_without_act(output, [1, 5, init_channel, init_channel], [init_channel], name='conv1')
    output = conv_without_act(output, [1, 1, init_channel, init_channel * 2], [init_channel * 2], name='conv2')
    output = conv_subsample_without_act(output, init_channel * 2, name='subsample')


  with tf.variable_scope('group3'):
    output = conv_act(output, [1, 5, init_channel * 2, init_channel * 2], [init_channel * 2], name='conv1')
    output = conv_act(output, [1, 1, init_channel * 2, init_channel * 4], [init_channel * 4], name='conv2')
    output = tf.nn.max_pool(output, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME', name='max_pool')


  with tf.variable_scope('group4'):
    output = conv_act(output, [1, 5, init_channel * 4, init_channel * 4], [init_channel * 4], name='conv1')
    output = conv_act(output, [1, 1, init_channel * 4, init_channel * 8], [init_channel * 8], name='conv2')
    output = tf.nn.max_pool(output, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME', name='max_pool')


  with tf.variable_scope('group5'):
    output = conv_act(output, [1, 5, init_channel * 8, init_channel * 8], [init_channel * 8], name='conv1')
    output = conv_act(output, [1, 1, init_channel * 8, init_channel * 16], [init_channel * 16], name='conv2')
    output = tf.nn.max_pool(output, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME', name='max_pool')

  with tf.variable_scope('group6'):
    output = conv_act(output, [1, 5, init_channel * 16, init_channel * 16], [init_channel * 16], name='conv1')
    output = conv_act(output, [1, 1, init_channel * 16, init_channel * 32], [init_channel * 32], name='conv2')
    output = tf.nn.max_pool(output, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME', name='max_pool')


  with tf.variable_scope('group7'):
    output = conv_act(output, [1, 5, init_channel * 32, init_channel * 32], [init_channel * 32], name='conv1')
    output = conv_act(output, [1, 1, init_channel * 32, init_channel * 64], [init_channel * 64], name='conv2')
    output = tf.nn.avg_pool(output, ksize=[1, 1, 250, 1], strides=[1, 1, 250, 1], padding='SAME', name='global_avg_pool')


  with tf.variable_scope('readout'):
    output = tf.reshape(output, [-1, init_channel * 64])

    w_fc = weight_variable([init_channel * 64, 2])
    b_fc = bias_variable([2])

    logits = tf.matmul(output, w_fc) + b_fc

  return logits

def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')

  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  weight_decay = 0.0
  l2_loss = tf.add_n(l2_loss_list, name='l2_loss')

  loss = cross_entropy_mean + l2_loss * weight_decay

  return loss

def train(loss):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  starter_learning_rate = 0.0001
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 1.0, staircase=True)

  train_op = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)

  return train_op, learning_rate, global_step

def evaluation(logits, labels):
  correct = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels)

  true_labels = tf.equal(labels, 1)

  true_positive = tf.logical_and(true_labels, correct)
  true_negative = tf.logical_and(tf.logical_not(true_labels), correct)
  false_positive = tf.logical_and(tf.logical_not(true_labels), tf.logical_not(correct))
  false_negative = tf.logical_and(true_labels, tf.logical_not(correct))

  correct_count = tf.reduce_sum(tf.cast(correct, tf.int32))

  tp_count = tf.reduce_sum(tf.cast(true_positive, tf.int32))
  tn_count = tf.reduce_sum(tf.cast(true_negative, tf.int32))
  fp_count = tf.reduce_sum(tf.cast(false_positive, tf.int32))
  fn_count = tf.reduce_sum(tf.cast(false_negative, tf.int32))

  return correct_count, tp_count, tn_count, fp_count, fn_count

def get_acc(correct_count, audios_placeholder, labels_placeholder, audios_data, labels_data):
  batch_size = 50

  true_count = 0

  steps_per_epoch = len(audios_data) // batch_size
  num_examples = steps_per_epoch * batch_size

  for step in xrange(steps_per_epoch):
    feed_dict={audios_placeholder: audios_data[step * batch_size: step * batch_size + batch_size],
      labels_placeholder: labels_data[step * batch_size: step * batch_size + batch_size]}
    true_count += correct_count.eval(feed_dict=feed_dict)

  accuracy = float(true_count) / num_examples

  # print('correct_num: {}, train_audios: {}'.format(true_count, num_examples))

  return accuracy


# def variable_init(saver, sess):
#   if 'load' in sys.argv:
#     load_file = '{}.ckpt'.format(os.path.basename(__file__).replace('.py', ''))
#     saver.restore(sess, './{}'.format(load_file))

#     print('load previous variable')
#   else:
#     sess.run(tf.initialize_all_variables())

def save_graph(sess):
  save_graph = os.path.basename(__file__).replace('.py', '_graph')
  train_writer = tf.train.SummaryWriter('./{}'.format(save_graph), sess.graph)
  train_writer.close()
  print('Save graph successfully')

def save_ckpt(saver, sess):
    save_file = '{}.ckpt'.format(os.path.basename(__file__).replace('.py', ''))
    saver.save(sess,  './{}'.format(save_file), write_meta_graph=False)
    print('Model saved in file: {}'.format(save_file))

def shuffle(data, labels):
  data_len = len(data)

  temp0 = np.arange(data_len / 2)
  np.random.shuffle(temp0)

  temp1 = np.reshape(temp0, (data_len / 2, 1))
  temp1 = temp1 * 2
  temp2 = temp1 + 1
  temp3 = np.concatenate((temp1, temp2), axis=1)
  perm = np.reshape(temp3, data_len)

  data = data[perm]
  labels = labels[perm]

def divide_train_set(train_audios, train_labels):
  proportion = 0.8

  shuffle(train_audios, train_labels)

  new_train_audios = train_audios[: int(len(train_audios) * 0.8)]
  new_train_labels = train_labels[: int(len(train_audios) * 0.8)]

  valid_audios = train_audios[int(len(train_audios) * 0.8) :]
  valid_labels = train_labels[int(len(train_audios) * 0.8) :]

  return new_train_audios, new_train_labels, valid_audios, valid_labels

def main():
  audios_placeholder = tf.placeholder(tf.float32, [None, 16000])
  labels_placeholder = tf.placeholder(tf.int32, [None])

  logits = inference(audios_placeholder)

  loss_op = loss(logits, labels_placeholder)

  train_op, lr_op, step_op = train(loss_op)

  correct_count, tp_count, tn_count, fp_count, fn_count = evaluation(logits, labels_placeholder)

  print('Construct model complete')

  sess = tf.InteractiveSession()

  saver = tf.train.Saver()


  # save_graph(sess)
  batch_size = 64

  # for i in range(100):
  valid_acc_list = []
  train_acc_list = []

  batch_index = 0

  train_audios_data, test_audios, train_labels_data, test_labels = read_data()
  train_audios, train_labels, valid_audios, valid_labels = divide_train_set(train_audios_data, train_labels_data)

  if 'load' in sys.argv:
    load_file = '{}.ckpt'.format(os.path.basename(__file__).replace('.py', ''))
    saver.restore(sess, './{}'.format(load_file))

    print('load previous variable')
  else:
    sess.run(tf.initialize_all_variables())

  for j in range(10000000):
    start = batch_index
    batch_index += batch_size

    if batch_index > len(train_audios):
      start = 0
      batch_index = batch_size

      shuffle(train_audios, train_labels)

    end = batch_index

    audios_batch = train_audios[start : end]
    labels_batch = train_labels[start : end]

    _, loss_value, lr, step = sess.run([train_op, loss_op, lr_op, step_op],
      feed_dict={audios_placeholder: audios_batch, labels_placeholder: labels_batch})

    # if step % 10 == 0:
    if step % 100 == 0:
      filename = os.path.basename(__file__)

      print('{} step {:d}, lr {:.8f}, loss {:.8f}'.format(filename, step, lr, loss_value))

    if step % 1000 == 0:
      print('-----')

      valid_accuracy = get_acc(correct_count, audios_placeholder, labels_placeholder, valid_audios, valid_labels)
      valid_acc_list.append(valid_accuracy)
      print('Valid accuracy: {:.8f}'.format(valid_accuracy))

      print('-----')

      train_accuracy = get_acc(correct_count, audios_placeholder, labels_placeholder, train_audios, train_labels)
      train_acc_list.append(train_accuracy)
      print('Train accuracy: {:.8f}'.format(train_accuracy))

      print('-----')

      save_ckpt(saver, sess)
      compute_time()

      print('-----')

    if step % 40000 ==0:
      print('Train complete')

      # filename = os.path.basename(__file__)

      # test_accuracy = get_acc(correct_count, audios_placeholder, labels_placeholder, test_audios, test_labels)
      # tp_accuracy = get_acc(tp_count, audios_placeholder, labels_placeholder, test_audios, test_labels)
      # tn_accuracy = get_acc(tn_count, audios_placeholder, labels_placeholder, test_audios, test_labels)
      # fp_accuracy = get_acc(fp_count, audios_placeholder, labels_placeholder, test_audios, test_labels)
      # fn_accuracy = get_acc(fn_count, audios_placeholder, labels_placeholder, test_audios, test_labels)
      # print('{} Test accuracy: {:.8f}'.format(filename, test_accuracy))
      # print('{} TP accuracy: {:.8f}'.format(filename, tp_accuracy))
      # print('{} TN accuracy: {:.8f}'.format(filename, tn_accuracy))
      # print('{} FP accuracy: {:.8f}'.format(filename, fp_accuracy))
      # print('{} FN accuracy: {:.8f}'.format(filename, fn_accuracy))

      # test_acc_list = [test_accuracy, tp_accuracy, tn_accuracy, fp_accuracy, fn_accuracy]

      # statistic_data_list = [valid_acc_list, train_acc_list, test_acc_list]

      # save_path = './{}/data{}.npy'.format(filename.replace('.py', ''), i)

      # np.save(save_path, statistic_data_list)

      # print('Save data to {}'.format(save_path))

      # print('-----')

      break


main()
