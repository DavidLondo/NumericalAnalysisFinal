import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def punto_fijo(g_input, x0, tol, iteramax=100):
    x = sp.symbols('x')
    
    try:
        if isinstance(g_input, str):
            g_sym = sp.sympify(g_input)
        elif callable(g_input):
            try:
                g_sym = sp.sympify(g_input(x))
            except Exception:
                g_sym = g_input
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
            g_prime_sym = sp.diff(g_sym, x)
            g_prime = sp.lambdify(x, g_prime_sym, modules=['numpy'])
        except Exception as e:
            return {"error": f"No se pudo calcular la derivada: {str(e)}"}

        puntos_prueba = np.linspace(x0 - tol*10, x0 + tol*10, 5)
        for p in puntos_prueba:
            try:
                val = g(p)
                if np.isnan(val) or np.isinf(val):
                    return {"error": f"La función g(x) no está definida en x = {p:.6f}"}
            except Exception:
                return {"error": f"La función g(x) no puede evaluarse en x = {p:.6f}"}

        advertencia = None
        try:
            gp_val = abs(g_prime(x0))
            if np.isnan(gp_val) or np.isinf(gp_val):
                advertencia = "No se pudo evaluar |g'(x0)|."
            elif gp_val >= 1:
                advertencia = f"Advertencia: |g'(x0)| = {gp_val:.4f} ≥ 1."
        except Exception:
            advertencia = "No se pudo evaluar g'(x0)."

        iteraciones = []
        xi_list = [x0]
        xi = x0

        for i in range(1, iteramax + 1):
            try:
                xi_new = g(xi)
                if np.isnan(xi_new) or np.isinf(xi_new):
                    return {
                        "error": f"g(x) no está definida en x = {xi:.6f}",
                        "iteraciones": iteraciones
                    }
                error = abs(xi_new - xi)
                iteraciones.append({
                    "iter": i,
                    "xi": xi,
                    "f_xi": xi_new,
                    "error": error if i > 1 else None
                })
                xi_list.append(xi_new)
                xi = xi_new
                if error < tol:
                    break
            except Exception as e:
                return {
                    "error": f"Error en iteración {i}: {str(e)}",
                    "iteraciones": iteraciones
                }

        else:
            return {
                "error": f"No se alcanzó la convergencia en {iteramax} iteraciones.",
                "iteraciones": iteraciones,
                "advertencia": advertencia
            }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        rango = max(5, abs(x0)*1.5)
        x_vals = np.linspace(x0 - rango, x0 + rango, 400)
        y_vals = g(x_vals)
        ax1.plot(x_vals, y_vals, label='g(x)', color='blue')
        ax1.plot(x_vals, x_vals, linestyle='--', color='gray', label='y = x')
        ax1.set_title('Método de Punto Fijo')
        ax1.set_xlabel('x')
        ax1.set_ylabel('g(x)')
        ax1.grid(True)
        for j in range(1, len(xi_list)):
            x_old = xi_list[j - 1]
            x_new = xi_list[j]
            ax1.plot([x_old, x_old], [x_old, x_new], 'r--')
            ax1.plot([x_old, x_new], [x_new, x_new], 'r--')
            ax1.plot(x_new, x_new, 'ro', markersize=4)
        ax1.legend()

        errores = [it['error'] for it in iteraciones if it['error'] is not None]
        if errores:
            ax2.semilogy(range(1, len(errores)+1), errores, 'g-', marker='o')
            ax2.set_title('Convergencia del Error')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Error (log)')
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
            "grafica_base64": image_base64,
            "advertencia": advertencia
        }

    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'cos(x)', '0.5*(x + 2/x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}
