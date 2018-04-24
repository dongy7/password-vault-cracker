import json

vsrc = 'data/vault.json'

with open(vsrc, 'r') as f:
    vaults = json.load(f)

out = 'data/vault.txt'
with open(out, 'w') as f:
    for key, vault in vaults.items():
        for pw in vault:
            if pw == '':
                print('Empty password')
            else:
                f.write(pw + '\n')
