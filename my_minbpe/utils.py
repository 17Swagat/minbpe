def getStats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, replace_id):
    i = 0
    newids = []
    while i < len(ids):
        if i == len(ids) - 1:
            newids.append(ids[i])
            break
        if (ids[i], ids[i+1]) == pair:
            newids.append(replace_id)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids