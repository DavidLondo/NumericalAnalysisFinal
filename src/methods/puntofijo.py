import sympy as sp
import math

def punto_fijo(g_str, x0, tol, iteramax=100):
    x = sp.symbols('x')
    try:
        g = sp.lambdify(x, g_str, modules=["math"])  # g(x) como string, usando funciones de math
    except Exception as e:
        return {"error": f"Error al interpretar g(x): {e}"}

    xi = x0
    iteraciones = []

    for i in range(1, iteramax + 1):
        try:
            xi_new = g(xi)
        except Exception as e:
            return {"error": f"Error al evaluar g(xi) en la iteración {i}: {e}"}

        error = abs(xi_new - xi)

        iteraciones.append({
            "iter": i,
            "xi": xi,
            "f_xi": xi_new,
            "error": error if i > 1 else None
        })

        if error < tol:
            return {
                "resultado": xi_new,
                "iteraciones": iteraciones
            }

        xi = xi_new

    return {
        "resultado": xi,
        "iteraciones": iteraciones
    }
