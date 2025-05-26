import numpy as np
from . import generar_grafica_base64

def vandermonde(x, y):
    n = len(x)
    A = np.vander(x, n)
    coef = np.linalg.solve(A, y)
    
    # Generate the graphical representation
    xpol = np.linspace(min(x), max(x), 200)
    ypol = np.polyval(coef, xpol)
    imagen = generar_grafica_base64(x, y, xpol, ypol, "Interpolación - Vandermonde")

    return {
        "solucion": coef,
        "grafica": imagen,
    }