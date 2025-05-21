import sympy as sp

def newton_raphson(fx_str, x0, tol, iteramax=100):
    x = sp.symbols('x')

    try:
        fx = sp.sympify(fx_str)
        dfx = sp.diff(fx, x)

        f = sp.lambdify(x, fx)
        df = sp.lambdify(x, dfx)
    except Exception as e:
        return {"error": f"Error al procesar la función: {e}"}

    iteraciones = []
    xi = x0
    error = None

    for i in range(1, iteramax + 1):
        try:
            fxi = f(xi)
            dfxi = df(xi)
        except Exception as e:
            return {"error": f"Error al evaluar la función en la iteración {i}: {e}"}

        if dfxi == 0:
            return {"error": f"La derivada se volvió cero en la iteración {i}. El método falla."}

        xi_new = xi - fxi / dfxi
        error = abs(xi_new - xi)

        iteraciones.append({
            "iter": i,
            "xi": xi,
            "f_xi": fxi,
            "error": error if i > 1 else None
        })

        if error < tol:
            break

        xi = xi_new

    return {
        "resultado": xi_new,
        "iteraciones": iteraciones
    }
