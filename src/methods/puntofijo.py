import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def punto_fijo(g_input, x0, tol, iteramax=100):
    """
    Método de Punto Fijo mejorado con gráfica incluida
    
    Args:
        g_input: Función (str, sympy o lambda)
        x0: Punto inicial
        tol: Tolerancia
        iteramax: Máximo iteraciones (default 100)
    """
    x = sp.symbols('x')
    
    try:
        if isinstance(g_input, str):
            g_sym = sp.sympify(g_input)
        elif callable(g_input):
            g_sym = g_input(x) if hasattr(g_input, '__call__') else g_input
        else:
            g_sym = g_input
            
        g = sp.lambdify(x, g_sym, modules=[
            'numpy',
            {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'ln': np.log, 'e': np.e, 'pi': np.pi,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'abs': np.abs, 'log10': np.log10
            }
        ])

        try:
            test_val = g(x0)
            if np.isnan(test_val):
                return {"error": f"g(x) no está definida en x0 = {x0}"}
        except Exception as e:
            return {"error": f"Error al evaluar g(x0): {str(e)}"}

        iteraciones = []
        xi = x0
        xi_list = [x0]

        for i in range(1, iteramax + 1):
            xi_new = g(xi)

            if np.isnan(xi_new):
                return {"error": f"g(x) no definida en x = {xi:.6f}",
                        "iteraciones": iteraciones}

            error = abs(xi_new - xi)

            iteraciones.append({
                "iter": i,
                "xi": xi,
                "f_xi": xi_new,
                "error": error if i > 1 else None
            })

            xi = xi_new
            xi_list.append(xi)

            if error < tol:
                break

        # ----------- GRÁFICA -----------
        fig, ax = plt.subplots()
        
        rango = max(5, abs(x0)*1.5)
        x_vals = np.linspace(x0 - rango, x0 + rango, 400)
        y_vals = g(x_vals)
        
        ax.plot(x_vals, y_vals, label='g(x)', color='blue')
        ax.plot(x_vals, x_vals, linestyle='--', color='gray', label='y = x')  # Línea identidad
        ax.set_title('Método de Punto Fijo')
        ax.set_xlabel('x')
        ax.set_ylabel('g(x)')
        ax.grid(True)

        for i in range(1, len(xi_list)):
            x_old = xi_list[i - 1]
            x_new = xi_list[i]
            ax.plot([x_old, x_old], [x_old, x_new], 'r--')
            ax.plot([x_old, x_new], [x_new, x_new], 'r--')
            ax.plot(x_new, x_new, 'ro', markersize=4)

        ax.legend()

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
        return {"error": "Expresión no válida. Ejemplos: 'cos(x)', '0.5*(x + 2/x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
