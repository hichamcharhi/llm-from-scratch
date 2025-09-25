import numpy as np

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.w1 = np.random.randn(dim, hidden_dim) * 0.01
        self.w2 = np.random.randn(hidden_dim, dim) * 0.01

    def __call__(self, x):
        return np.dot(np.maximum(0, np.dot(x, self.w1)), self.w2)

class MultiHeadAttention:
    def __init__(self, n_embd, n_head):
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = np.random.randn(n_embd, 3 * n_embd) * 0.01
        self.c_proj = np.random.randn(n_embd, n_embd) * 0.01

    def __call__(self, x):
        B, T, C = x.shape
        qkv = np.dot(x, self.c_attn)
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        mask = np.tril(np.ones((T, T)))
        att = att - 1e9 * (1 - mask)
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)
        y = np.matmul(att, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return np.dot(y, self.c_proj)

class TransformerBlock:
    def __init__(self, n_embd, n_head):
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, 4 * n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

if __name__ == "__main__":
    n_embd, n_head, block_size = 64, 4, 16
    block = TransformerBlock(n_embd, n_head)
    x = np.random.randn(1, block_size, n_embd)
    out = block(x)
    print("Forme après TransformerBlock :", out.shape)
