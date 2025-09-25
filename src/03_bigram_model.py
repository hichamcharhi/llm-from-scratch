import numpy as np
from collections import defaultdict

# Données (simulées ici)
text = "abracadabra"
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Créer mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder le texte
data = [stoi[c] for c in text]

# Initialiser la matrice de comptage
N = np.zeros((vocab_size, vocab_size), dtype=np.int32)

# Compter les bigrams
for i in range(len(data) - 1):
    N[data[i], data[i+1]] += 1

# Convertir en probabilités (avec lissage de Laplace)
P = (N + 1).astype(np.float32)
P /= P.sum(axis=1, keepdims=True)

# Génération
def generate(n_tokens, start_ix=0):
    out = []
    ix = start_ix
    for _ in range(n_tokens):
        p = P[ix]
        ix = np.random.choice(vocab_size, p=p)
        out.append(itos[ix])
    return ''.join(out)

if __name__ == "__main__":
    print("Génération :", generate(20))
