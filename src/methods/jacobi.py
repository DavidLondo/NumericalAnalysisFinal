import numpy as np

def jacobi(A, b, x0=None, tol=1e-5, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)

    history = []
    for i in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        history.append((i, x_new.tolist(), error))
        if error < tol:
            break
        x = x_new
    return {
        "solucion": x.tolist(),
        "iteraciones": history
    }

def calcular_radio_espectral(A):
    eigvals = np.linalg.eigvals(A)
    return max(abs(eigvals))

def analizar_convergencia(A, metodo, w=1.1):
    A = np.array(A, dtype=float)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    if metodo == "jacobi":
        D_inv = np.linalg.inv(D)
        B = D_inv @ (L + U)

    elif metodo == "gaussseidel":
        DL_inv = np.linalg.inv(D + L)
        B = DL_inv @ U

    elif metodo == "sor":
        DL = D + w * L
        B = np.linalg.inv(DL) @ ((1 - w) * D - w * U)

    else:
        return None, False  # método desconocido

    radio = max(abs(np.linalg.eigvals(B)))
    puede_converger = radio < 1
    return radio, puede_converger