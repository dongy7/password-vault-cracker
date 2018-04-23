import json
import random
import re
from kullback_leibler import calculate_divergence


def construct_decoy_set(vsize, decoys, size):
    random.shuffle(decoys)
    # return [decoy[:vsize] for decoy in decoys[:size]]
    return [decoy[:50] for decoy in decoys[:size]]


def get_vaults(path, vsize):
    vaults = []
    i = 0

    vault = []
    with open(dv_path, 'r') as f:
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


def get_dist(vault, real=False):
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


vpath = 'data/vault.json'

with open(vpath, 'r') as f:
    vault = json.load(f)

print('reading password vaults')
vaults = sorted(
    [val for _, val in vault.items()], key=lambda x: len(x), reverse=True)

groups = ['2-3', '4-8', '9-50']
vault_groups = {}

for g in groups:
    vault_groups[g] = {}
    vault_groups[g]['vaults'] = []
    vault_groups[g]['dists'] = []

for vault in vaults:
    if len(vault) < 2:
        continue
    elif len(vault) <= 3:
        group = '2-3'
    elif len(vault) <= 8:
        group = '4-8'
    else:
        group = '9-50'
    vault_groups[group]['vaults'].append(vault)
    vault_groups[group]['dists'].append(get_dist(vault, True))

# for v in vaults:
#     print(v)

print('constructing decoy vaults')
dv_path = 'data/decoy_vaults.txt'
d_vaults = get_vaults(dv_path, 50)

print('constructing probability distribution of decoys')
pw_dist = get_pw_dist('data/rockyou-withcount.txt')

# print(d_vaults[0])
print('Average rank')
for g in vault_groups:
    ranks = []
    for dist in vault_groups[g]['dists']:
        decoys = construct_decoy_set(len(dist) - 1, d_vaults, 999)
        decoy_dists = [get_dist(v) for v in decoys]
        test_set = [dist] + decoy_dists
        for cv in test_set:
            cv['___score___'] = calculate_divergence(cv, pw_dist)

        sorted_set = sorted(
            test_set, key=lambda x: x['___score___'], reverse=True)
        ranks.append(rank(sorted_set))

    avg = sum(ranks) / len(ranks) / 1000 * 100
    print(g + ': ' + str(round(avg, 2)) + '%')
