from collections import defaultdict
import random

# Corpus d'exemple
corpus = "le chat mange le poisson le chien mange la viande".split()

# Construire le modèle Bigram
bigrams = defaultdict(list)
for i in range(len(corpus) - 1):
    bigrams[corpus[i]].append(corpus[i + 1])

# Générer une phrase
def generate(start_word, length=5):
    current = start_word
    output = [current]
    for _ in range(length - 1):
        if current in bigrams:
            next_word = random.choice(bigrams[current])
            output.append(next_word)
            current = next_word
        else:
            break
    return " ".join(output)

if __name__ == "__main__":
    print(generate("le"))  # Ex: "le chat mange le poisson"
