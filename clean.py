import sys
import collections

input = sys.argv[1]
output = sys.argv[2]

max_len = 20

counts = collections.defaultdict(int)
with open(input, 'r') as f:
    for line in f:
        password = line.rstrip()
        if len(password) > max_len:
            continue
        counts[password] += 1

sorted_counts = sorted([(pw, count) for (pw, count) in counts.items()], key=lambda x: x[1], reverse=True)

with open(output, 'w') as f:
    for (pw, count) in sorted_counts:
        line = '%7d %s\n' % (count, pw)
        f.write(line)