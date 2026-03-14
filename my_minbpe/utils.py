def getStats(ids: list) -> dict:
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    i = 0
    new_ids = []
    while i < len(ids): # - 1:
        if (i+1) >= len(ids):
            new_ids += [ids[i]]
            i += 1
            continue
        if (ids[i], ids[i+1]) == pair:
            new_ids += [idx]
            i += 2
        else:
            new_ids += [ids[i]]
            i += 1
    return new_ids
