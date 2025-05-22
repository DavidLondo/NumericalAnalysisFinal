import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-5, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    history = []
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        error = np.linalg.norm(x_new - x, ord=np.inf)
        history.append((k, x_new.tolist(), error))
        if error < tol:
            break
        x = x_new

    return {
        "solucion": x.tolist(),
        "iteraciones": history
    }