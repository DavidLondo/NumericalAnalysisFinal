import sympy as sp

def regla_falsa(fx_str, a, b, tol, iteramax=100):
    x = sp.symbols('x')
    try:
        # Convertir la expresión a función evaluable
        expr = sp.sympify(fx_str)
        fx = sp.lambdify(x, expr, modules=["numpy"])
        fa = fx(a)
        fb = fx(b)
    except Exception as e:
        return {"error": f"Error al evaluar la función: {e}"}

    if fa * fb >= 0:
        return {"error": "El método de la regla falsa no es aplicable. La función no cambia de signo en el intervalo."}

    iteraciones = []
    c = a  # inicializamos
    c_ant = None
    error = None
    itera = 1

    while itera <= iteramax:
        c = b - fb * (a - b) / (fa - fb)
        try:
            fc = fx(c)
        except Exception as e:
            return {"error": f"Error al evaluar la función en c={c}: {e}"}

        if c_ant is not None:
            error = abs(c - c_ant)
            if error < tol:
                # Ya cumplimos la tolerancia, salimos
                iteraciones.append({
                    "iter": itera,
                    "a": a,
                    "b": b,
                    "c": c,
                    "f_a": fa,
                    "f_b": fb,
                    "f_c": fc,
                    "error": error
                })
                break
        else:
            error = None

        iteraciones.append({
            "iter": itera,
            "a": a,
            "b": b,
            "c": c,
            "f_a": fa,
            "f_b": fb,
            "f_c": fc,
            "error": error
        })

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_ant = c
        itera += 1

    return {
        "resultado": c,
        "iteraciones": iteraciones
    }
