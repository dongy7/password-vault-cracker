from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import csv,os,sys
import json


# on multi-GPU server, tensorflow trends to using all available GPUs
# so we need to limit the number of GPU that it can use
# the number_tag of GPU can be seen by typing command "nvidia-smi"
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # onely using GPU in this list, e.g. "1,4,6"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# usage percentile, if it is 1, it means using 100% of this GPU
# if it is 0.5, it means using 50% of this GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.85
set_session(tf.Session(config=config))


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
from RNN_utils import *

# Read config options
with open('config.json', 'r') as f:
  config_data = json.load(f)

# Parsing arguments for Network definition

basedir = os.path.dirname(os.path.abspath(__file__))
dataset = config_data['dataset']
model_dir = config_data['model_dir']
model = config_data['model']

ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default=os.path.join(basedir, config_data['data_dir'], dataset))
ap.add_argument('-batch_size', type=int, default=64)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-seq_length', type=int, default=20)
ap.add_argument('-hidden_dim', type=int, default=640)
ap.add_argument('-generate_length', type=int, default=20)
ap.add_argument('-nb_epoch', type=int, default=40)

# default='' means generation; default='train' means training
ap.add_argument('-mode', default='')

ap.add_argument('-train', action='store_true')
ap.add_argument('-load', action='store_true')

# if you don't want to load previously trained model,
# remember delete the ending letters ".hdf5"
# ap.add_argument('-weights', default= basedir + '/trained_model/' +
#               'checkpoint_layer_3_hidden_640_epoch_9_dataset_rockyou.hdf5')
ap.add_argument('-weights', default=os.path.join(basedir, model_dir, model))

args = vars(ap.parse_args())
print(args)

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# this flag is used in load_data(), as some parts of load_data()
# is time-consuming and is only necessary in training mode
train_flag = args['train']
# if args['mode'] == 'train':
#   train_flag = 1
# else:
#   train_flag = 0

# Creating training data
time_start = time.time()
X, y, i2c, c2i = load_data(DATA_DIR, SEQ_LENGTH, train_flag)
VOCAB_SIZE = len(i2c.keys())

time_end = time.time()
print('Training data construction time: %0.1f Minutes' % ((time_end - time_start) / 60))

# Creating and compiling the Network
print('Constructing Model...')
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Generate some sample before training to know how bad it is!
#generate_text(model, args['generate_length'], VOCAB_SIZE, i2c)

nb_epoch = 0
# check if you want to load model from previously trained model
# if you don't want to load model, remember delelte the last few letters
# when you initialize WEIGHTS in ap.add_argument
if args['load']:
  print('Loading data from trained model...\n')
  model.load_weights(WEIGHTS)
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('epoch') + 6:WEIGHTS.find('dataset') - 1])

epoch_interval = config_data['epoch_interval']

# Training
# if args['mode'] == 'train':
if train_flag:
  print('Training...\n')
  while True:
    time_start = time.time()
    print('Epoch: {}\n'.format(nb_epoch))
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, i2c)
    time_end = time.time()
    print('Training time spent for this epoch: %0.1f Minutes'%((time_end-time_start)/60))
    if nb_epoch % epoch_interval == 0: # save model at how many epochs
      model.save_weights(basedir +'/trained_model/checkpoint_layer_{}_hidden_{}_epoch_{}_dataset_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch,dataset.split('-')[0]))
      print('model saved: '+basedir +'/checkpoint_layer_{}_hidden_{}_epoch_{}_dataset_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch,dataset.split('-')[0]))


# Else, performing generation
else:
  for i in range(50):
    pw = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, i2c)