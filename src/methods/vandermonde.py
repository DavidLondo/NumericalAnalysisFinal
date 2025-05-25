import numpy as np
from . import generar_grafica_base64

def vandermonde(x, y):
    n = len(x)
    A = np.vander(x, n)
    coef = np.linalg.solve(A, y)
    # Transform coef to string eg. a*x^3 + b*x^2 + c*x + d
    str_coef = [f"{coef[i]}*x^{n-i-1}" for i in range(n) if coef[i] != 0]
    #remove x^1 and x^0
    str_coef = [s.replace("*x^1", "*x") for s in str_coef]
    str_coef = [s.replace("*x^0", "") for s in str_coef]
    for i in range(len(str_coef)-1):
        if not str_coef[i+1].startswith("-"):
            str_coef[i] += " +"
    
    # Generate the graphical representation
    xpol = np.linspace(min(x), max(x), 200)
    ypol = np.polyval(coef, xpol)
    imagen = generar_grafica_base64(x, y, xpol, ypol, "Interpolación - Vandermonde")

    return {
        "solucion": " ".join(str_coef),
        "grafica": imagen,
    }