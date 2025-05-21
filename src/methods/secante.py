import sympy as sp
import numpy as np

def secante(fx_str, X0, X1, tol, iteramax=100):
    x = sp.symbols('x')
    try:
        expr = sp.sympify(fx_str)
        fx = sp.lambdify(x, expr, modules=["numpy"])
        fX0 = fx(X0)
        fX1 = fx(X1)
    except Exception as e:
        return {"error": f"Error al evaluar la función: {e}"}

    xn = [X0, X1]
    fn = [fX0, fX1]
    iteraciones = []
    c = 0
    error = None

    # Registrar primera iteración (sin error aún)
    iteraciones.append({
        "iter": c,
        "Xi": xn[1],
        "f_Xi": fn[1],
        "error": None
    })

    while error is None or (error >= tol and abs(fn[-1]) > 1e-15) and c < iteramax:
        try:
            # Método de la secante
            x_new = xn[-1] - fn[-1] * (xn[-1] - xn[-2]) / (fn[-1] - fn[-2])
            f_new = fx(x_new)
        except Exception as e:
            return {"error": f"Error en iteración {c+1}: {e}"}

        xn.append(x_new)
        fn.append(f_new)
        c += 1
        error = abs(xn[-1] - xn[-2])

        iteraciones.append({
            "iter": c,
            "Xi": xn[-1],
            "f_Xi": fn[-1],
            "error": error
        })

        # Si la función evalúa a 0, podemos parar
        if abs(f_new) < 1e-15:
            break

    return {
        "resultado": xn[-1],
        "iteraciones": iteraciones
    }
