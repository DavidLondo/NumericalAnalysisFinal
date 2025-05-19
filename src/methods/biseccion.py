import sympy as sp

def biseccion(fx, a, b, tol):
    x = sp.symbols('x')
    f = sp.lambdify(x, fx)

    resultados = []  # para almacenar resultados de cada iteración
    
    if f(a) * f(b) >= 0:
        return {"error": "El método de bisección no es aplicable. La función no cambia de signo en el intervalo."}
    
    i = 1
    error = None
    x_ant = None

    while abs(b - a) > tol:
        xi = (a + b) / 2
        f_xi = f(xi)

        if x_ant is not None:
            error = abs(xi - x_ant)
        else:
            error = None

        resultados.append({
            "iter": i,
            "xi": xi,
            "f_xi": f_xi,
            "error": error
        })

        if f_xi == 0:
            break
        elif f(a) * f_xi < 0:
            b = xi
        else:
            a = xi

        x_ant = xi
        i += 1

    return {
        "resultado": xi,
        "iteraciones": resultados
    }
