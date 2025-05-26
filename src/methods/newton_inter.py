import numpy as np
from . import generar_grafica_base64

def newtoninter(x, y):
    n = len(x)
    tabla = np.zeros((n, n+1))

    for i in range(n):
        tabla[i, 0] = x[i]
        tabla[i, 1] = y[i]

    for j in range(2, n + 1):
        for i in range(j-1, n):
            tabla[i, j] = (tabla[i, j-1] - tabla[i-1, j-1]) / (tabla[i, 0] - tabla[i-j+1, 0])

    coef = [tabla[i, i + 1] for i in range(n)]

    pol = np.array([0.0])
    acum = np.array([1.0])

    for i in range(n):
        term = coef[i] * acum
        if len(term) > len(pol):
            pol = np.pad(pol, (len(term) - len(pol), 0))
        elif len(pol) > len(term):
            term = np.pad(term, (len(pol) - len(term), 0))
        pol = pol + term
        if i < n - 1:
            acum = np.convolve(acum, [1, -x[i]])

    # Transform coef to string eg. a*x^3 + b*x^2 + c*x + d
    str_coef = [f"{pol[i]}*x^{n-i-1}" for i in range(n) if pol[i] != 0]
    # remove x^1 and x^0
    str_coef = [s.replace("*x^1", "*x") for s in str_coef]
    str_coef = [s.replace("*x^0", "") for s in str_coef]
    for i in range(len(str_coef)-1):
        if not str_coef[i+1].startswith("-"):
            str_coef[i] += " +"

    # Generate the graphical representation
    xpol = np.linspace(min(x), max(x), 200)
    ypol = np.polyval(pol, xpol)
    imagen = generar_grafica_base64(x, y, xpol, ypol, "Interpolación - Newton")

    return {
        "solucion": " ".join(str_coef),
        "grafica": imagen,
        "tabla": tabla.tolist(),
    }