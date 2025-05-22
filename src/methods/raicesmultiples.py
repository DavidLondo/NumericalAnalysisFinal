import math

def raices_multiples(Fun, derivada_fx1, derivada_fx2, X0, tol, Niter=100):
    x = X0
    f = eval(Fun)
    derivada1 = eval(derivada_fx1)
    derivada2 = eval(derivada_fx2)
    denominador = derivada1**2 - (f * derivada2)
    Error = 100
    c = 0
    xi = x

    iteraciones = [{
        "iter": c,
        "Xi": xi,
        "f_Xi": f,
        "error": None
    }]

    while Error > tol and f != 0 and denominador != 0 and c < Niter:
        x = x - (f * derivada1) / denominador
        xi_new = x
        f = eval(Fun)
        derivada1 = eval(derivada_fx1)
        derivada2 = eval(derivada_fx2)
        denominador = derivada1**2 - (f * derivada2)
        Error = abs(xi_new - xi)
        c += 1
        xi = xi_new

        iteraciones.append({
            "iter": c,
            "Xi": xi,
            "f_Xi": f,
            "error": Error
        })

    return {
        "resultado": xi,
        "iteraciones": iteraciones
    }
