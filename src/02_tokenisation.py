from collections import defaultdict
import re

# Tokenizer caractère simple
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int to string

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

# BPE simplifié
def get_stats(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

if __name__ == "__main__":
    # Exemple caractère
    text = "abracadabra"
    tokenizer = CharTokenizer(text)
    print("Encodage 'abra':", tokenizer.encode("abra"))
    print("Décodage:", tokenizer.decode([0, 1, 3, 0]))

    # Exemple BPE
    text = "bonjour bonjour"
    tokens = list(text.encode("utf-8"))
    vocab_size = 260
    num_merges = vocab_size - 256
    merges = {}
    for i in range(num_merges):
        stats = get_stats(tokens)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        tokens = merge(tokens, pair, idx)
        merges[pair] = idx
    print("Fusions effectuées :", len(merges))
