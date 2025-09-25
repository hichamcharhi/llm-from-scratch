import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = np.dot(weights, V)
    return output

if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_k, d_v = 4, 8, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    out = scaled_dot_product_attention(Q, K, V)
    print("Forme de la sortie :", out.shape)  # (4, 8)
