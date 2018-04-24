import json
import random
import re
from kullback_leibler import calculate_divergence
from RNN_utils import pw_loss_calc

"""
vsize: number of passwords in each vault
decoys: list of decoy password vaults
size: number of decoy vaults

returns list of decoy vaults that have vsize number of passwords each
"""
def construct_decoy_set(vsize, decoys, size):
    random.shuffle(decoys)
    return [[pw for pw in decoy if len(pw) > 0][:vsize] for decoy in decoys[:size]]
    # return [vault[:vsize] for vault in decoys[:size]]

"""
path: path to file with vault passwords
vsize: number of passwords in each vault
returns list of decoy vaults each containing vsize pws
"""
def get_vaults(path, vsize):
    vaults = []
    i = 0

    vault = []
    with open(path, 'r') as f:
        for line in f:
            i += 1
            pw = line.strip()
            vault.append(pw)

            if i % vsize == 0:
                vaults.append(vault)
                vault = []

    return vaults


"""
pw_file: pw file where each line is: count password
returns dictionary with number and proability for each pw in file
"""
def get_pw_dist(pw_file):
    # matching: 12345 password
    r = re.compile('\s+(\d+)\s([^\\n]+)')
    # matching empty password
    r2 = re.compile('\s+(\d+)')
    dist = {}
    sum = 0
    with open(pw_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            m = r.match(line)
            try:
                count, pw = (m.group(1), m.group(2))
                sum += int(count)
            except:
                m = r2.match(line)
                count, pw = (m.group(1), '')
            dist[pw] = {}
            dist[pw]['occ'] = int(count)
            dist[pw]['prob'] = 0

    for pw in dist:
        dist[pw]['prob'] = dist[pw]['occ'] / sum
    return dist

"""
vault: list of passwords in vault
real: True if vault is not a decoy vault
model: model parameters if using loss for KL divergence
returns a dictionary containing the probability and occurence for each pw in vault
"""
def get_dist(vault, real=False, model=None):
    dist = {}
    for pw in vault:
        if pw not in dist:
            dist[pw] = {}
            dist[pw]['prob'] = 0
            dist[pw]['occ'] = 1
        else:
            dist[pw]['occ'] += 1

    for pw in dist:
        dist[pw]['prob'] = dist[pw]['occ'] / len(vault)

    if model:
        for pw in dist:
            if len(pw) == 0:
                raise Exception
            dist[pw]['score'] = pw_loss_calc(model['model'], model['SEQ_LENGTH'], model['VOCAB_SIZE'], model['i2c'], model['c2i'], pw)

    dist['___+ToTaL+___'] = {}
    dist['___+ToTaL+___']["sum"] = len(vault)
    dist['___real___'] = real

    return dist

"""
test_set: list of vaults in order of KL divergence in decreasing order
returns the rank of the real vault
"""
def rank(test_set):
    i = 0
    for cv in test_set:
        if cv['___real___']:
            return i
        i += 1

"""
group: size of password vaults one of [2-3, 4-8, 9-50]
model: parameters for pw model specified if using loss from neural network
ranks the password vaults in a set of 1000 vaults
"""
def eval_KL(group='2-3', model=None):
    # reading vault data
    print('Ranking vaults of size: ' + group)

    out_file = 'results/group_{}.json'.format(group)
    vpath = 'data/vault.json'

    with open(vpath, 'r') as f:
        vault = json.load(f)

    print('reading password vaults')
    vaults = sorted(
        [[pw for pw in val if len(pw) > 0] for _, val in vault.items()], key=lambda x: len(x), reverse=True)

    dists = []
    lo, hi = [int(x) for x in group.split('-')]

    # computing distribution for each vault in the group
    for vault in vaults:
        if len(vault) >= lo and len(vault) <= hi:
            dists.append(get_dist(vault, True, model))

    print('constructing decoy vaults')
    dv_path = 'data/decoy_vaults.txt'
    d_vaults = get_vaults(dv_path, 50)

    print('constructing probability distribution of decoys')
    pw_dist = get_pw_dist('data/decoys_withcount.txt')

    j = 0
    ranks = []
    ranks_wloss = []
    print('Saving output to {}'.format(out_file))

    # for each real vault, rank it in a set of 1000 vaults containing 999 decoy vaults
    for dist in dists:
        j += 1
        print('Vault {}/{}'.format(j, len(dists)))
        decoys = construct_decoy_set(len(dist) - 2, d_vaults, 999)
        decoy_dists = [get_dist(v, False, model) for v in decoys]
        test_set = [dist] + decoy_dists

        # compute score with and without loss
        for cv in test_set:
            cv['___score___'] = calculate_divergence(cv, pw_dist)
            cv['___score_with_loss___'] = calculate_divergence(cv, pw_dist, True)

        # sort by the score in descending order
        sorted_set = sorted(
            test_set, key=lambda x: x['___score___'], reverse=True)
        sorted_loss_set = sorted(
            test_set, key=lambda x: x['___score_with_loss___'], reverse=True)

        # find the rank of the real vault
        v_rank = rank(sorted_set)
        v_rank_wloss = rank(sorted_loss_set)

        print("Rank: {},{}".format(v_rank, v_rank_wloss))

        # append rank of real vault
        ranks.append(v_rank)
        ranks_wloss.append(v_rank_wloss)

    # find average rank across all real vaults
    avg = sum(ranks) / len(ranks) / 1000 * 100
    avg_wloss = sum(ranks_wloss) / len(ranks_wloss) / 1000 * 100
    data = {
        'avg_rank': avg,
        'ranks': ranks,
        'ranks_wloss': ranks_wloss,
        'avg_rank_wloss': avg_wloss
    }

    with open(out_file, 'w') as f:
        json.dump(data, f)