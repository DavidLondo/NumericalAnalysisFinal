import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify
import io
import base64

def biseccion(fx, a, b, tol):
    x = sp.symbols('x')
    
    try:
        f_sym = sp.sympify(fx)
        f = sp.lambdify(x, f_sym, modules=[
            'numpy',
            {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
             'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
             'ln': np.log, 'e': np.e, 'pi': np.pi}
        ])
        
        resultados = []

        fa, fb = f(a), f(b)
        if np.isnan(fa) or np.isnan(fb):
            return {"error": "La función no está definida en los extremos del intervalo"}
        if fa * fb >= 0:
            return {"error": "El método no es aplicable. La función no cambia de signo en el intervalo."}

        i = 1
        error = None
        x_ant = None
        xi_list = []

        while abs(b - a) > tol:
            xi = (a + b) / 2
            f_xi = f(xi)

            if np.isnan(f_xi):
                return {"error": f"La función no está definida en x = {xi:.6f}", "iteraciones": resultados}

            if x_ant is not None:
                error = abs(xi - x_ant)
            else:
                error = None

            resultados.append({
                "iter": i,
                "xi": xi,
                "f_xi": f_xi,
                "error": error
            })
            xi_list.append(xi)

            if f_xi == 0:
                break
            elif f(a) * f_xi < 0:
                b = xi
            else:
                a = xi

            x_ant = xi
            i += 1

        # ----------- GENERAR GRÁFICA -----------
        fig, ax = plt.subplots()
        x_vals = np.linspace(float(a)-1, float(b)+1, 400)
        y_vals = f(x_vals)

        ax.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(xi, color='red', linestyle='--', label='Raíz Aproximada')
        ax.set_title('Gráfica de f(x) y aproximación con Bisección')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)

        # Marcar los xi de cada iteración
        for xi_i in xi_list:
            ax.axvline(xi_i, color='orange', linestyle=':', linewidth=0.8)

        # Convertir la figura a base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": xi,
            "iteraciones": resultados,
            "grafica_base64": image_base64
        }

    except sp.SympifyError:
        return {"error": "Expresión matemática inválida. Use formato como 'cos(x) + x**2'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
