import json

vsrc = 'data/vault.json'

with open(vsrc, 'r') as f:
    vaults = json.load(f)

out = 'data/vault-clean.txt'
with open(out, 'w') as f:
    for key, vault in vaults.items():
        for pw in vault:
            f.write(pw + '\n')
        f.write('---\n')