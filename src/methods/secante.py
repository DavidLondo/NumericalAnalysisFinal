import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def secante(fx_input, X0, X1, tol, iteramax=100):
    """
    Método de la Secante con gráfica base64
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
            f0, f1 = f(X0), f(X1)
            if np.isnan(f0) or np.isnan(f1):
                return {"error": "La función no está definida en los puntos iniciales"}
        except Exception as e:
            return {"error": f"Error al evaluar función en puntos iniciales: {str(e)}"}

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
                    return {
                        "resultado": xn[-1],
                        "iteraciones": iteraciones,
                        "warning": "Divisor cercano a cero. Posible convergencia lenta."
                    }
                
                x_new = xn[-1] - fn[-1] * (xn[-1] - xn[-2]) / denominator
                f_new = f(x_new)

                if np.isnan(f_new):
                    return {
                        "resultado": xn[-1],
                        "iteraciones": iteraciones,
                        "error": f"Función no definida en x = {x_new:.6f}"
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
                    "resultado": xn[-1],
                    "iteraciones": iteraciones,
                    "error": f"Error en iteración {c}: {str(e)}"
                }

        # ----------- GRÁFICA -----------
        fig, ax = plt.subplots()
        ax.plot(xn[2:], fn[2:], 'bo-', label='f(Xn)')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title("Método de la Secante - f(Xn)")
        ax.set_xlabel("Xn")
        ax.set_ylabel("f(Xn)")
        ax.grid(True)
        ax.legend()

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
        return {"error": "Expresión no válida. Ejemplos: 'x*tan(x)-1', 'exp(-x)-cos(x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
