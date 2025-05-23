import numpy as np

def sor(A, b, w=1.1, x0=None, tol=1e-5, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    history = []
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] if j < i else A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (1 - w) * x[i] + (w / A[i][i]) * (b[i] - sigma)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        history.append((k, x_new.tolist(), error))
        if error < tol:
            break
        x = x_new

    return {
        "solucion": x.tolist(),
        "iteraciones": history
    }