import numpy as np
from . import generar_grafica_base64

def lagrange(x, y):
    n = len(x)
    pol = np.zeros(n)

    def Li(i):
        numerador = np.poly1d([1.0])
        denominador = 1.0
        for j in range(n):
            if j != i:
                numerador *= np.poly1d([1.0, -x[j]])  # (x - xj)
                denominador *= (x[i] - x[j])
        return numerador / denominador

    resultado = np.poly1d([0.0])
    for i in range(n):
        resultado += y[i] * Li(i)

    coef = resultado.coefficients

    # Generate the graphical representation
    xpol = np.linspace(min(x), max(x), 200)
    ypol = resultado(xpol)
    imagen = generar_grafica_base64(x, y, xpol, ypol, "Interpolación - Lagrange")

    return {
        "solucion": coef,
        "grafica": imagen,
    }
