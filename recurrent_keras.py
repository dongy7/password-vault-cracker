from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import csv, os, sys
import json
import random

# on multi-GPU server, tensorflow trends to using all available GPUs
# so we need to limit the number of GPU that it can use
# the number_tag of GPU can be seen by typing command "nvidia-smi"

# os.environ["CUDA_VISIBLE_DEVICES"]="3" # onely using GPU in this list, e.g. "1,4,6"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# usage percentile, if it is 1, it means using 100% of this GPU
# if it is 0.5, it means using 50% of this GPU
# config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
from RNN_utils import *
from keras.callbacks import ModelCheckpoint, History
from vault_utils import *

# Parsing arguments for Network definition
# data = 'rockyou'
data = 'decoys'
basedir = os.path.dirname(os.path.abspath(__file__))
dataset = data + '_withcount.txt.bz2'
# dataset = data + '-withcount.txt.bz2'
num = '10'
loss = '0.3978'
val_loss = '1.5480'
# num = '03'
# loss = '0.5601'
# val_loss = '0.9915'

# decoys_L_3_H_320_epoch_10_loss_0.3978_val_loss_1.5480.hdf5

ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default=basedir + '/data/' + dataset)
ap.add_argument('-batch_size', type=int, default=64)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-seq_length', type=int, default=20)
ap.add_argument('-hidden_dim', type=int, default=320)
ap.add_argument('-generate_length', type=int, default=20)
ap.add_argument('-nb_epoch', type=int, default=40)
ap.add_argument('-mode', default='')
ap.add_argument('-group', default='2-3')
args = vars(ap.parse_args())
# default='' means generation; default='train' means training
# if you don't want to load previously trained model,
# remember delete the ending letters ".hdf5"
ap.add_argument(
    '-weights',
    default=basedir + '/trained_model/history1/' + data + '_L_' +
    str(args['layer_num']) + '_H_' + str(args['hidden_dim']) + '_epoch_' +
    num + '_loss_' + loss + '_val_loss_' + val_loss + '.hdf5')

args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']
group_size = args['group']

# this flag is used in load_data(), as some parts of load_data()
# is time-consuming and is only necessary in training mode
if args['mode'] == 'train':
    train_flag = 1
else:
    train_flag = 0

# Creating training data
time_start = time.time()
X, y, i2c, c2i = load_data(DATA_DIR, SEQ_LENGTH, train_flag)
VOCAB_SIZE = len(i2c.keys())

#np.random.shuffle(X) # shuflle data set

time_end = time.time()
print('Training data construction time: %0.1f Minutes' %
      ((time_end - time_start) / 60))

# Creating and compiling the Network
print('Constructing Model...')
model = Sequential()
model.add(
    LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
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
if WEIGHTS[-5:] == '.hdf5':
    print('Loading data from trained model...\nModel = ', WEIGHTS, '\n')
    model.load_weights(WEIGHTS)
    nb_epoch = int(
        WEIGHTS[WEIGHTS.rfind('epoch') + 6:WEIGHTS.find('loss') - 1])

# Training
if args['mode'] == 'train':
    print('Training...\n')

    filepath = WEIGHTS[:WEIGHTS.rfind(
        'epoch'
    )] + 'epoch_{epoch:02d}_loss_{loss:.4f}_val_loss_{val_loss:.4f}.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=filepath,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        period=1)
    model.fit(
        X,
        y,
        validation_split=0.1,
        batch_size=BATCH_SIZE,
        verbose=1,
        nb_epoch=20,
        callbacks=[checkpointer])
elif args['mode'] == 'eval':
    from vault_utils import eval_KL
    print('evaluating KL divergence score')
    params = {
        'model': model,
        'SEQ_LENGTH': SEQ_LENGTH,
        'GENERATE_LENGTH': GENERATE_LENGTH,
        'VOCAB_SIZE': VOCAB_SIZE,
        'i2c': i2c,
        'c2i': c2i
    }
    eval_KL(group_size, params)
elif args['mode'] == 'prep':
    from preprocess import prep_vault
    print('preprocessing decoy vault')
    params = {
        'model': model,
        'SEQ_LENGTH': SEQ_LENGTH,
        'GENERATE_LENGTH': GENERATE_LENGTH,
        'VOCAB_SIZE': VOCAB_SIZE,
        'i2c': i2c,
        'c2i': c2i
    }
    prep_vault(params, 'data/decoy_vaults.txt', 'data/decoy_scores.json')
    # prep_vault(params, 'data/decoy_vaults.txt', 'data/real_scores.json')

# Else, performing generation
else:
    # for i in range(10):
    # pw = generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, i2c)

    test_pw(model, SEQ_LENGTH, VOCAB_SIZE, i2c, c2i)
    #beamsearch_text( model, GENERATE_LENGTH, VOCAB_SIZE, i2c, c2i)