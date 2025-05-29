import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def regla_falsa(fx_input, a, b, tol, iteramax=100):
    x = sp.symbols('x')
    
    try:
        if isinstance(fx_input, str):
            fx_str = fx_input.replace('^', '**').replace('sen', 'sin')
            f_expr = sp.sympify(fx_str)
        else:
            f_expr = fx_input

        f = sp.lambdify(x, f_expr, modules=[
            'numpy',
            {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'ln': np.log, 'log10': np.log10, 'abs': np.abs,
                'e': np.e, 'pi': np.pi
            }
        ])

        try:
            fa, fb = f(a), f(b)
            if np.isnan(fa) or np.isnan(fb):
                return {"error": "La función no está definida en los extremos del intervalo."}
            if fa * fb >= 0:
                return {"error": "El método no es aplicable. La función no cambia de signo en [a, b]."}
        except Exception as e:
            return {"error": f"Error al evaluar función en los extremos: {str(e)}"}

        try:
            x_vals = np.linspace(a, b, 100)
            f_vals = f(x_vals)
            if np.any(np.isnan(f_vals)) or np.any(np.isinf(f_vals)):
                return {"error": "La función presenta discontinuidades o singularidades en [a, b]."}
        except Exception as e:
            return {"error": f"Error al verificar continuidad: {str(e)}"}

        iteraciones = []
        c_vals = []
        c_ant = None

        for itera in range(1, iteramax + 1):
            try:
                if (fa - fb) == 0:
                    return {
                        "error": "División por cero. f(a) = f(b), no se puede continuar.",
                        "iteraciones": iteraciones,
                        "ultimo_valor": c_ant
                    }

                c = b - fb * (a - b) / (fa - fb)
                fc = f(c)

                if np.isnan(fc) or np.isinf(fc):
                    return {
                        "error": f"La función no está definida en c = {c:.6f}",
                        "iteraciones": iteraciones,
                        "ultimo_valor": c
                    }

                c_vals.append(c)
                error = abs(c - c_ant) if c_ant is not None else None

                iteraciones.append({
                    "iter": itera,
                    "a": a, "b": b, "c": c,
                    "f_a": fa, "f_b": fb, "f_c": fc,
                    "error": error
                })

                if error is not None and error < tol:
                    break

                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc

                c_ant = c

            except ZeroDivisionError:
                return {
                    "error": "División por cero detectada. La función puede ser constante.",
                    "iteraciones": iteraciones,
                    "ultimo_valor": c_ant
                }
            except Exception as e:
                return {
                    "error": f"Error en iteración {itera}: {str(e)}",
                    "iteraciones": iteraciones,
                    "ultimo_valor": c_ant
                }

        if error is None or error >= tol:
            return {
                "error": f"No se alcanzó la convergencia en {iteramax} iteraciones. Último error: {error:.6f}",
                "ultimo_valor": c,
                "iteraciones_realizadas": itera,
                "iteraciones": iteraciones
            }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(c_vals, [f(ci) for ci in c_vals], 'ro-', label='f(c)')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_title("Evolución de f(c)")
        ax1.set_xlabel("c")
        ax1.set_ylabel("f(c)")
        ax1.grid(True)
        ax1.legend()

        x_plot = np.linspace(a, b, 100)
        ax2.plot(x_plot, f(x_plot), 'b-', label='f(x)')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.plot(c_vals, [f(ci) for ci in c_vals], 'ro', label='Aproximaciones')
        ax2.set_title("Función en el intervalo [a, b]")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        grafica_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": c,
            "iteraciones": iteraciones,
            "grafica_base64": grafica_base64,
            "iteraciones_totales": itera
        }

    except sp.SympifyError:
        return {"error": "Expresión matemática inválida. Use formato como 'cos(x) + x**2'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
