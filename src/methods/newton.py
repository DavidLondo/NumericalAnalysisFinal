import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def newton_raphson(fx_str, x0, tol, iteramax=100):
    """
    Método de Newton-Raphson mejorado con generación de gráfica
    
    Args:
        fx_str: Función como string (ej: "cos(x)*exp(-x)")
        x0: Punto inicial
        tol: Tolerancia
        iteramax: Máximo de iteraciones (default 100)
    """
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

        try:
            test_val = f(x0)
            if np.isnan(test_val):
                return {"error": f"La función no está definida en x0 = {x0}"}
        except Exception as e:
            return {"error": f"Error al evaluar función en x0: {str(e)}"}

        iteraciones = []
        xi = x0
        xi_list = []

        for i in range(1, iteramax + 1):
            fxi = f(xi)
            dfxi = df(xi)

            if np.isnan(fxi) or np.isnan(dfxi):
                return {"error": f"Función o derivada no definida en iteración {i}, x = {xi:.6f}",
                        "iteraciones": iteraciones}

            if dfxi == 0:
                return {"error": f"Derivada cero en iteración {i}. Método falló.",
                        "iteraciones": iteraciones}

            xi_new = xi - fxi / dfxi
            error = abs(xi_new - xi)

            iteraciones.append({
                "iter": i,
                "xi": xi,
                "f_xi": fxi,
                "df_xi": dfxi,
                "error": error if i > 1 else None
            })
            xi_list.append(xi)

            if error < tol:
                break

            xi = xi_new

        # ----------- GENERAR GRÁFICA -----------
        fig, ax = plt.subplots()
        
        # Rango centrado en x0 (más +/- 5 unidades)
        rango = max(5, abs(x0)*1.5)
        x_vals = np.linspace(x0 - rango, x0 + rango, 400)
        y_vals = f(x_vals)
        
        ax.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(xi, color='red', linestyle='--', label='Raíz Aproximada')
        ax.set_title('Gráfica de f(x) con Newton-Raphson')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)

        # Puntos de las iteraciones
        for xi_i in xi_list:
            ax.plot(xi_i, f(xi_i), 'ro', markersize=4)

        ax.legend()

        # Convertir a imagen base64
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
