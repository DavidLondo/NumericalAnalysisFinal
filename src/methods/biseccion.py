import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify
import io
import base64

def biseccion(fx, a, b, tol, iteramax=100):
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
        xi_list = []
        errores = []

        # Verificación de continuidad y valores definidos
        puntos_prueba = np.linspace(a, b, 50)
        for punto in puntos_prueba:
            try:
                val = f(punto)
                if not np.isfinite(val):
                    return {"error": f"La función no está definida o no es continua en x = {punto:.6f}"}
            except Exception as e:
                return {"error": f"La función no es continua o tiene problemas en x = {punto:.6f}: {e}"}

        # Verificación inicial
        fa, fb = f(a), f(b)
        if fa * fb >= 0:
            return {"error": "El método no es aplicable. La función no cambia de signo en el intervalo [a, b]."}

        i = 1
        error = None
        x_ant = None

        while i <= iteramax:
            xi = (a + b) / 2
            try:
                f_xi = f(xi)
            except Exception as e:
                return {"error": f"La función no está definida en x = {xi:.6f}"}

            # Calcular error relativo
            if x_ant is not None:
                error = abs(xi - x_ant)
                errores.append(error)
            else:
                error = None

            resultados.append({
                "iter": i,
                "xi": xi,
                "f_xi": f_xi,
                "error": error
            })
            xi_list.append(xi)

            # Criterio de parada
            if f_xi == 0 or (error is not None and error < tol):
                break

            # Actualizar intervalo
            if f(a) * f_xi < 0:
                b = xi
            else:
                a = xi

            x_ant = xi
            i += 1

        # ----------- GENERAR GRÁFICAS -----------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Gráfica de f(x) y aproximación
        x_vals = np.linspace(float(a)-1, float(b)+1, 400)
        try:
            y_vals = f(x_vals)
            ax1.plot(x_vals, y_vals, label='f(x)', color='blue')
        except Exception:
            ax1.text(0.5, 0.5, "No se pudo graficar f(x)", horizontalalignment='center')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(xi, color='red', linestyle='--', label='Raíz Aproximada')
        ax1.set_title('f(x) y raíz con Bisección')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True)
        for xi_i in xi_list:
            ax1.axvline(xi_i, color='orange', linestyle=':', linewidth=0.8)

        # Gráfica de convergencia del error (logarítmica)
        if errores:
            ax2.semilogy(range(1, len(errores)+1), errores, 'g-', marker='o')
            ax2.set_title('Convergencia del Error')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Error (escala log)')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "No hay errores calculados para graficar", horizontalalignment='center')
            ax2.set_axis_off()

        plt.tight_layout()

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
        return {"error": f"Ocurrió un error inesperado: {str(e)}"}
