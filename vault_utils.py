import json
import random
import re
from kullback_leibler import calculate_divergence
from RNN_utils import pw_loss_calc

def construct_decoy_set(vsize, decoys, size):
    random.shuffle(decoys)
    return [[pw for pw in decoy if len(pw) > 0][:vsize] for decoy in decoys[:size]]
    # return [vault[:vsize] for vault in decoys[:size]]

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


def get_pw_dist(pw_file):
    r = re.compile('\s+(\d+)\s([^\\n]+)')
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


def rank(test_set):
    i = 0
    for cv in test_set:
        if cv['___real___']:
            return i
        i += 1

def eval_KL(group='2-3', model=None):
    print('Ranking vaults of size: ' + group)
    vpath = 'data/vault.json'

    with open(vpath, 'r') as f:
        vault = json.load(f)

    print('reading password vaults')
    vaults = sorted(
        [[pw for pw in val if len(pw) > 0] for _, val in vault.items()], key=lambda x: len(x), reverse=True)

    # groups = ['2-3', '4-8', '9-50']
    # vault_groups = {}
    #     vault_groups[g]['vaults'] = []
    #     vault_groups[g]['dists'] = []

    dists = []
    lo, hi = [int(x) for x in group.split('-')]

    for vault in vaults:
        if len(vault) >= lo and len(vault) <= hi:
            vaults.append(vault)
            dists.append(get_dist(vault, True, model))

        # if len(vault) < 2:
        #     continue
        # elif len(vault) <= 3:
        #     group = '2-3'
        # elif len(vault) <= 8:
        #     group = '4-8'
        # else:
        #     group = '9-50'
        # vault_groups[group]['vaults'].append(vault)
        # vault_groups[group]['dists'].append(get_dist(vault, True, model))

    # for v in vaults:
    #     print(v)

    print('constructing decoy vaults')
    dv_path = 'data/decoy_vaults.txt'
    d_vaults = get_vaults(dv_path, 50)

    print('constructing probability distribution of decoys')
    # pw_dist = get_pw_dist('data/rockyou-withcount.txt')
    pw_dist = get_pw_dist('data/decoys_withcount.txt')

    j = 0
    ranks = []
    # print(d_vaults[0])
    print(len(dists))
    for dist in dists:
        j += 1
        print('Vault ' + str(j) + '/' + str(len(dists)))
        decoys = construct_decoy_set(len(dist) - 2, d_vaults, 999)
        # print(decoys)
        decoy_dists = [get_dist(v, False, model) for v in decoys]
        test_set = [dist] + decoy_dists
        for cv in test_set:
            cv['___score___'] = calculate_divergence(cv, pw_dist)

        sorted_set = sorted(
            test_set, key=lambda x: x['___score___'], reverse=True)
        v_rank = rank(sorted_set)
        print("Rank: " + str(v_rank))
        ranks.append(v_rank)

    avg = sum(ranks) / len(ranks) / 1000 * 100
    print('Average rank for ' + g + ': ' + str(round(avg, 2)) + '%')
