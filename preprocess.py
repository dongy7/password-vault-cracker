import json
import time
from RNN_utils import pw_loss_calc

"""
model: parameters for model
vault_src: file path of decoy password vault
dest: output file path
precomputes scores for passwords in decoy vaults using trained model and stores dictionary in json file
"""
def prep_vault(model, vault_src, dest):
    scores = {}
    with open(vault_src, 'r') as f:
        i = 0
        for line in f:
            i += 1
            print('{}/{}'.format(i, 1007300))
            pw = line.strip()
            if pw == '':
                continue
            if pw in scores:
                continue
            scores[pw] = pw_loss_calc(model['model'], model['SEQ_LENGTH'], model['VOCAB_SIZE'], model['i2c'], model['c2i'], pw)

    with open(dest, 'w') as f:
        json.dump(scores, f)
