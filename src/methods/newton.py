import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def newton_raphson(fx_str, x0, tol, iteramax=100):
    x = sp.symbols('x')
    
    try:
        fx = sp.sympify(fx_str)
        dfx = sp.diff(fx, x)

        f = sp.lambdify(x, fx, modules=[
            'numpy',
            {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'ln': np.log, 'e': np.e, 'pi': np.pi,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'sec': lambda x: 1/np.cos(x), 'csc': lambda x: 1/np.sin(x)
            }
        ])
        df = sp.lambdify(x, dfx, modules='numpy')

        test_points = [x0, x0 + tol, x0 - tol]
        for pt in test_points:
            val_f = f(pt)
            val_df = df(pt)
            if np.isnan(val_f) or np.isnan(val_df):
                return {"error": f"La función o su derivada no está definida cerca de x = {pt:.6f}"}

        iteraciones = []
        xi = x0

        for i in range(1, iteramax + 1):
            fxi = f(xi)
            dfxi = df(xi)

            if np.isnan(fxi) or np.isnan(dfxi):
                return {"error": f"Función o derivada no definida en iteración {i}, x = {xi:.6f}"}

            if abs(dfxi) < 1e-12:
                return {"error": f"Derivada cercana a cero en iteración {i}. Método no puede continuar."}

            xi_new = xi - fxi / dfxi
            error = abs(xi_new - xi)

            iteraciones.append({
                "iter": i,
                "xi": xi,
                "f_xi": fxi,
                "error": error if i > 1 else None
            })

            if error < tol:
                xi = xi_new
                break

            xi = xi_new
        else:
            return {"error": f"No se alcanzó la convergencia en {iteramax} iteraciones."}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        rango = max(5, abs(x0)*1.5)
        x_vals = np.linspace(x0 - rango, x0 + rango, 400)
        y_vals = f(x_vals)

        ax1.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(xi, color='red', linestyle='--', label='Raíz Aproximada')
        ax1.set_title('Función y aproximaciones')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.grid(True)

        for it in iteraciones:
            ax1.plot(it["xi"], it["f_xi"], 'ro', markersize=4)
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
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": xi,
            "iteraciones": iteraciones,
            "grafica_base64": image_base64
        }

    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'x*tan(x)-1', 'exp(-x)-cos(x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
