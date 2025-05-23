import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def regla_falsa(fx_input, a, b, tol, iteramax=100):
    """
    Método de Regla Falsa con gráfica base64
    """
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
                return {"error": "La función no está definida en los extremos del intervalo"}
            if fa * fb >= 0:
                return {"error": "El método no es aplicable. La función no cambia de signo en el intervalo."}
        except Exception as e:
            return {"error": f"Error al evaluar la función: {str(e)}"}

        iteraciones = []
        c_ant = None
        c_vals = []

        for itera in range(1, iteramax + 1):
            try:
                c = b - fb * (a - b) / (fa - fb)
                fc = f(c)
                c_vals.append(c)

                if np.isnan(fc):
                    return {"error": f"La función no está definida en c = {c:.6f}",
                            "iteraciones": iteraciones}
                
                error = abs(c - c_ant) if c_ant is not None else None
                
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
                
                if error is not None and error < tol:
                    break

                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    
                c_ant = c

            except ZeroDivisionError:
                return {"error": "División por cero. La función puede ser constante.",
                        "iteraciones": iteraciones}
            except Exception as e:
                return {"error": f"Error en iteración {itera}: {str(e)}",
                        "iteraciones": iteraciones}

        # ----------- GRÁFICA -----------
        y_vals = [f(ci) for ci in c_vals]

        fig, ax = plt.subplots()
        ax.plot(c_vals, y_vals, 'ro-', label='f(c)')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title("Regla Falsa - Evolución de f(c)")
        ax.set_xlabel("c")
        ax.set_ylabel("f(c)")
        ax.grid(True)
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        grafica_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": c,
            "iteraciones": iteraciones,
            "grafica_base64": grafica_base64
        }
        
    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'cos(x)-x', 'exp(-x)-log(x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
