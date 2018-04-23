from __future__ import print_function
import numpy as np
import bz2
import string
from keras.preprocessing import sequence
import random



def print_history(history):
    # print the history log from training process
    # print(history)
    # I have incorporate this function into Keras package in
    # callback.py class:ModelCheckpoint function:on_epoch_end
    loss = history['loss']
    try:
        val_loss = history['val_loss']
    except:
        print('there is no validation data')
        val_loss = loss
    print('Training History:')
    for i in range(len(loss)):
        print('Epoch_%d: loss=%.4f; val_loss=%.4f'%(i,loss[i],val_loss[i]))



def format_input(input,gen_length, vocab_size):
    X = np.zeros((input.shape[0], gen_length, vocab_size))
    return  input

def show(name,data):
    # for dubug, show type,shape and value of one variable
    if isinstance(data,list) == True:
        print('\n{}: type={},shape={}, =\n{}'.format(name,type(data), np.asarray(data).shape, data))
        return
    if isinstance(data,np.ndarray) == True:
        print('\n{}: type={},shape={}, =\n{}'.format(name, type(data), data.shape, data))
        return
    print('show error: {} is not string or list'.format(name))
    return


def beam_get_sentence( fin_trans,i2c ):
    # used for beam search, after beam search finishes
    # the result is in labels, need change them into vocabulary
    ps = []
    for idx, e in enumerate(fin_trans):
        sentence = ''
        for i in range(1,len(e)-1):
            sentence += i2c[e[i]]
        ps.append(sentence)
    return ps



# method for generating text
def beamsearch_text(model, gen_length, vocab_size, i2c, c2i):

    # starting with random character
    k = 20
    beam_size = k

    fin_trans = []
    fin_costs = []
    fin_align = []

    trans = [[]]*beam_size
    costs = np.zeros((beam_size,1))
    n_samples = beam_size

    inp = np.array([c2i['<S>']] * beam_size)
    #print('inp: type={},shape={}, ={}'.format(type(inp),inp.shape,inp))

    X = np.zeros((beam_size, gen_length, vocab_size))
    X[:, 0, :][:, c2i['<S>'] ] = 1
    predict = model.predict(X[:, :1, :])
    output = predict[:, -1, :]

    log_probs = np.log(output[0, :])

    inp = inp[:,None]
    for i in range( 1, gen_length ):

        next_costs = np.array([0.0])[:, None] - log_probs
        flat_next_costs = next_costs.flatten()
        best_costs_indices = np.argpartition(next_costs.flatten(), n_samples)[:n_samples]

        best_costs_indices = best_costs_indices[:,None]
        inp = np.concatenate((inp,best_costs_indices),axis=1)

        trans_indices = [int(idx) for idx in best_costs_indices / vocab_size]  # which beam line
        word_indices = best_costs_indices % vocab_size
        best_costs = flat_next_costs[best_costs_indices]
        costs = np.add(costs[trans_indices],best_costs)

        ### Filter the sequences that end with end-of-sequence character ###
        inp[:,i-1] = (inp[:,i-1])[trans_indices]
        inp[:,i] = word_indices.T

        delete_idx = []
        for idx,e in enumerate(word_indices):
            if e == c2i['<EOS>']:
                fin_costs.append(costs.tolist()[idx])
                fin_trans.append(inp.tolist()[idx])
                delete_idx.append(idx)
                n_samples -= 1
                if n_samples == 0:
                    break
        inp = np.delete(inp,delete_idx,0)
        costs = np.delete(costs,delete_idx,0)

        if n_samples == 0:
            break

        ### Preparing input and predict next output  ###
        X = np.zeros((n_samples, gen_length, vocab_size))
        for idx in range(n_samples):
            for j in range(i+1):
                X[idx,j,inp[idx,j]] = 1

        predict = model.predict(X[:, :i+1, :])

        output = predict[:, -1, :]
        log_probs = np.log(output)

    print(beam_get_sentence(fin_trans, i2c))
    return fin_trans, fin_costs


# method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
    # this is with the original source code, generate with greedy algorithm
	# starting with random character
	ix = [np.random.randint(vocab_size)]
	ix = [1] # starting from <S> sample, c2i["<S>"] = 1
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocab_size))
	for i in range(length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)

		if ix[-1] in [0,1,2]: # if
			break
		y_char.append(ix_to_char[ix[-1]])
	pw = ('').join(y_char)
	print('pw={}'.format(pw))
	return pw


def test_pw(model, SEQ_LENGTH, VOCAB_SIZE, i2c, c2i):
    # maybe you need to completely rewrite this function

    flag = 1
    if flag == 0:
        # test decoy
        f = open('data/decoy.txt','r')
    else:
        # test real
        f = bz2.open('data/decoys-test.txt.bz2','r')
        # f = bz2.open('data/rockyou-test.txt.bz2','r')

    loss_total = []
    for n, line in enumerate(f):

        nonchar_flag = 0
        if flag == 1: # if real, the data is not clean, first need to clean
            try:
                # the .bz2 file is stored in bytes, decode is necessary
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                print("Cannot decode: {}".format(line))
                continue

            line = line.strip().split() # delete space and line-change symbol
            line = line[1]

            for c in line:
                if c not in string.printable[:94]:
                    nonchar_flag = 1
                    break

            if nonchar_flag == 1:
                continue
        # end of clean real password dataset


        line = line.strip()
        if n > 200: # only tested the top 200 passwords
            break

        loss = pw_loss_calc(model, SEQ_LENGTH, VOCAB_SIZE, i2c, c2i, line)
        loss_total.append(loss)

    # average loss for the whole test set
    print( sum(loss_total)/len(loss_total) )

    # save result in txt file
    fout = open('score.txt','w')
    for line in loss_total:
        fout.write(str(line)[:5])
        fout.write('\n')
    fout.close()




def pw_loss_calc( model, length, vocab_size, i2c, c2i, pw):
    # input one password, return its loss
    # pw to be tested should only contain the pw part
    # <S> and <EOS> should not be included in pw

    X = np.zeros((1,length+1,vocab_size))
    X[0,0,:][c2i['<S>']] = 1 # the first character is <S>
    loss = 0
    for i in range(length):

        # if i exceeds the length of pw
        if i >= len(pw):
            break
        try:
            idx = c2i[pw[i]]
        except:
            print('error   ',pw)
            return 0
        predict = model.predict(X[:,:i+1,:])[0][0]
        temp_loss = 0 - np.log(predict[idx])
        loss += temp_loss
        X[0,i+1,:][idx] = 1 # append this char to X

    return loss/len(pw) # loss per character

# method for preparing the training data
def load_data(data_dir, seq_length, train_flag):

    print('\nConstructing training set...')
    f = bz2.open(data_dir, 'r')
    dataset = []
    unique_pw = 0
    for n, line in enumerate(f):

        if n > 5 and 0:
            break

        try:
            # the .bz2 file is stored in bytes, decode is necessary
            line = line.decode('utf-8')
        except UnicodeDecodeError:
            print("Cannot decode: {}".format(line))
            continue
        line = line.strip().split() # delete space and line-change symbol

        if len(line) > 1 and line[0].isdigit():
            # the dataset is stored in the format: frequency word
            # e.g. 200 123456; 100 password
            # so, line[0] is freq, line[1] is pw

            freq_threshold = 10

            if int(line[0]) <= freq_threshold:
                # ignore passwords whose freqency is less than the threshold
                print('Passwords with freqency less than '+str(freq_threshold)+' are ignored...')
                break

            # if there is non-printable char in this pw, then ignore this pw
            nonchar_flag = 0
            for c in line[1]:
                if c not in string.printable[:94]:
                    nonchar_flag = 1
                    break

            # repeat password N times, N is the frequency
            if nonchar_flag == 0:
                unique_pw += 1
                for i in range(int(int(line[0])/freq_threshold)):
                    # because in raw data, the highese freq is very large
                    # e.g. freq("123456") is more than 10,000 times, so
                    # we divide the actual freq by 10, otherwise too large to train
                    dataset.append(line[1]) # repeat freq/10 times

    # get vocab, must sort, otherwise the order of chars change everytime
    # chars = sorted(list(set(''.join(dataset)))) # get unique characters in dataset
    chars = [c for c in string.printable] # using printable as vocabulary
    chars = ['<PAD>','<S>','<EOS>','<UN>'] + chars # add special symbols
    VOCAB_SIZE = len(chars)

    print('Dataset: {} passwords, unique {}'.format(len(dataset),unique_pw))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

    i2c = {ix:char for ix, char in enumerate(chars)}
    c2i = {char:ix for ix, char in enumerate(chars)}

    # because the rest processing is time-consuming and not necessary for generation
    # we skip the rest part if we are not in training mode
    if train_flag == 0:
        X = np.zeros((1, seq_length, VOCAB_SIZE))
        Y = np.zeros((1, seq_length, VOCAB_SIZE))
        return X, Y, i2c, c2i

    # add <s> and <EOS> to dataset
    data_list = []
    for n, line in enumerate(dataset):

        data_list.append([])
        data_list[n].append(c2i['<S>']) # each pw starts with <S>
        for c in line:
            if c in c2i:
                data_list[n].append(c2i[c])
            else:
                data_list[n].append(c2i['<UN>']) #
            if len(data_list[n]) >= seq_length:
                break
        data_list[n].append(c2i['<EOS>'])

    # get x, y
    x = np.asarray( [sent[:-1] for sent in data_list])
    y = np.asarray( [sent[1:] for sent in data_list])

    # pading
    for n,line in enumerate(x):
        if len(line) < seq_length:
            x[n].extend( [c2i['<PAD>'] for i in range(seq_length - len(line))] )
    for n,line in enumerate(y):
        if len(line) < seq_length:
            y[n].extend( [c2i['<PAD>'] for i in range(seq_length - len(line))] )

    # vectoring x,y
    X = np.zeros((len(data_list), seq_length, VOCAB_SIZE))
    Y = np.zeros((len(data_list), seq_length, VOCAB_SIZE))
    for i in range(0, len(data_list)):

        if i%50000 == 0: # showing progress
            print('Vectorize training data: {} out of {}'.format(i,len(data_list)))

        # using one-hot encoding
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            input_sequence[j][x[i][j]] = 1.
            target_sequence[j][y[i][j]] = 1.
        X[i] = input_sequence
        Y[i] = target_sequence

    return X, Y, i2c, c2i





















####################################
#### below are junk functions ######
####################################




def clear_data():
    # this function divide the raw dataset into
    # training/validation sets

    val_ratio = 0.1 # data ratio as validation set
    data = 'yahoo-withcount'
    data_dir = 'data/' + data + '.txt.bz2'
    filetype = data_dir.split('.')[-1]
    if filetype == 'bz2':
        f = bz2.open(data_dir,'r')
    elif filetype == 'txt':
        f = open(data_dir,'r')
    else:
        print('Datatype not supported...\n')
        exit(1)

    cumsum = []
    sum = 0
    dataset = []
    for n,line in enumerate(f):

        if filetype == 'bz2':
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                print("Cannot decode: {}".format(line))
                continue
        line = line.strip().split()  # delete space and line-change symbol

        if len(line) > 1 and line[0].isdigit():
            sum += int(line[0])  # total sum until now
            cumsum.append(sum)  # cumsum

            dataset.append([int(line[0]),line[1]])

    # generate random numbers between [0,sum]
    val_index = [random.randint(0,sum) for r in range(int(val_ratio*sum))]

    # from val_index to actual password
    val = [0]*len(cumsum)
    for n,idx in enumerate(val_index):
        if n % 100 == 0:
            print('n=',n)
        # if val_index is the first password
        if idx < cumsum[0]:
            val[0] += 1
            continue

        # if val_index is the final password
        elif idx >= cumsum[-2]:
            val[-1] += 1
            continue

        # else
        for i in range(1,len(cumsum)):
            if idx < cumsum[i]:
                val[i] += 1
                break

    # store data into files
    train_f = open('data/'+data+'_training_set.txt','w')
    val_f = open('data/'+data+'_validation_set.txt','w')
    for n,line in enumerate(dataset):
        train_f.write(str(line[0] - val[n]))
        train_f.write(' ' + line[1] + '\n')
        val_f.write(str(val[n]))
        val_f.write(' ' + line[1] +'\n')


    #print('\n\ncumsum = \n{}\n\nval_index=\n{}\n\n'.format(cumsum, val_index))
    #print('val=\n{}\n\ndataset=\n{}\n\n'.format(val,dataset))
    exit(1)




