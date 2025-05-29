import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def secante(fx_input, X0, X1, tol, iteramax=100):
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
            f0, f1 = f(X0), f(X1)
            if np.isnan(f0) or np.isnan(f1):
                return {
                    "error": "La función no está definida en los puntos iniciales.",
                    "iteraciones": []
                }
        except Exception as e:
            return {
                "error": f"Error al evaluar función en puntos iniciales: {str(e)}",
                "iteraciones": []
            }

        xn = [float(X0), float(X1)]
        fn = [f0, f1]
        iteraciones = []

        iteraciones.append({
            "iter": 0,
            "Xi": xn[1],
            "f_Xi": fn[1],
            "error": None
        })

        for c in range(1, iteramax + 1):
            try:
                denominator = (fn[-1] - fn[-2])
                if abs(denominator) < 1e-15:
                    break

                x_new = xn[-1] - fn[-1] * (xn[-1] - xn[-2]) / denominator
                f_new = f(x_new)

                if np.isnan(f_new) or np.isinf(f_new):
                    return {
                        "error": f"La función no está definida en x = {x_new:.6f}.",
                        "iteraciones": iteraciones
                    }

                xn.append(x_new)
                fn.append(f_new)
                error = abs(xn[-1] - xn[-2])

                iteraciones.append({
                    "iter": c,
                    "Xi": xn[-1],
                    "f_Xi": fn[-1],
                    "error": error
                })

                if error < tol or abs(f_new) < 1e-12:
                    break

            except Exception as e:
                return {
                    "error": f"Error en iteración {c}: {str(e)}",
                    "iteraciones": iteraciones
                }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(xn[2:], fn[2:], 'bo-', label='f(Xn)')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_title("Evolución de f(Xn)")
        ax1.set_xlabel("Xn")
        ax1.set_ylabel("f(Xn)")
        ax1.grid(True)
        ax1.legend()

        errores = [it['error'] for it in iteraciones if it['error'] is not None]
        if errores:
            ax2.semilogy(range(1, len(errores)+1), errores, 'g-', marker='o')
            ax2.set_title('Convergencia del Error')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Error (escala log)')
            ax2.grid(True)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        grafica_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": xn[-1],
            "iteraciones": iteraciones,
            "grafica_base64": grafica_base64
        }

    except sp.SympifyError:
        return {
            "error": "Expresión no válida. Ejemplos: 'x*tan(x)-1', 'exp(-x)-cos(x)'",
            "iteraciones": []
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "iteraciones": []
        }
