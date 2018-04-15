from __future__ import print_function
import numpy as np
import bz2
import string
from keras.preprocessing import sequence


# method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
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


# method for preparing the training data
def load_data(data_dir, seq_length, train_flag):

    print('\nConstructing training set...')
    f = bz2.open(data_dir, 'r')
    dataset = []
    for n, line in enumerate(f):
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
            if int(line[0]) <= 10:
                # ignore passwords whose freqency is less than the threshold
                print('Passwords with freqency less than 10 are ignored...')
                break

            # if there is non-printable char in this pw, then ignore this pw
            nonchar_flag = 0
            for c in line[1]:
                if c not in string.printable[:94]:
                    nonchar_flag = 1
                    break

            # repeat password N times, N is the frequency
            if nonchar_flag == 0:
                for i in range(int(int(line[0])/10)):
                    # because in raw data, the highese freq is very large
                    # e.g. freq("123456") is more than 10,000 times, so
                    # we divide the actual freq by 10, otherwise too large to train
                    dataset.append(line[1]) # repeat freq/10 times

    # get vocab, must sort, otherwise the order of chars change everytime
    chars = sorted(list(set(''.join(dataset)))) # get unique characters in dataset
    chars = ['<PAD>','<S>','<EOS>','<UN>'] + chars # add special symbols
    VOCAB_SIZE = len(chars)

    print('Dataset: {} passwords'.format(len(dataset)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

    i2c = {ix:char for ix, char in enumerate(chars)}
    c2i = {char:ix for ix, char in enumerate(chars)}

    # because the rest processing is time-consuming and not necessary for generation
    # we skip the rest part if we are not in training mode
    if not train_flag:
        X = np.zeros((1, seq_length, VOCAB_SIZE))
        Y = np.zeros((1, seq_length, VOCAB_SIZE))
        return X, Y, i2c, c2i

    # add <s> and <EOS> to dataset
    data_list = []
    for n, line in enumerate(dataset):

        # if line contains non-printable char
        for c in line:
            if c not in string.printable[:94]:
                continue

        data_list.append([])
        data_list[n].append(c2i['<S>'])
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