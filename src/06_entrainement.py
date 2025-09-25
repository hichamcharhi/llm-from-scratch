import numpy as np

def cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)
    logits -= np.max(logits, axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(B * T), targets])
    return loss

def get_batch(data, batch_size, block_size):
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

if __name__ == "__main__":
    # Données simulées
    data = np.random.randint(0, 1000, size=10000)
    vocab_size = 1000

    # Simulation de logits
    x, y = get_batch(data, batch_size=4, block_size=16)
    logits = np.random.randn(4, 16, vocab_size)

    loss = cross_entropy_loss(logits, y)
    print(f"Perte simulée : {loss:.4f}")
