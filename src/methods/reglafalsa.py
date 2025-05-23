import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def regla_falsa(fx_input, a, b, tol, iteramax=100):
    """
    Método de Regla Falsa mejorado con manejo de convergencia
    
    Args:
        fx_input: Función como string o expresión SymPy
        a: Extremo izquierdo del intervalo
        b: Extremo derecho del intervalo
        tol: Tolerancia para el criterio de parada
        iteramax: Máximo número de iteraciones permitidas
    
    Returns:
        Diccionario con:
        - resultado: si converge
        - iteraciones: tabla de iteraciones
        - grafica_base64: gráfica en base64 (si converge)
        - error: mensaje si no converge o hay error
        - ultimo_valor: última aproximación calculada
    """
    x = sp.symbols('x')
    
    try:
        # Preprocesamiento de la función
        if isinstance(fx_input, str):
            fx_str = fx_input.replace('^', '**').replace('sen', 'sin')
            f_expr = sp.sympify(fx_str)
        else:
            f_expr = fx_input

        # Creación de función numérica
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

        # Validación inicial del intervalo
        try:
            fa, fb = f(a), f(b)
            if np.isnan(fa) or np.isnan(fb):
                return {"error": "La función no está definida en los extremos del intervalo."}
            if fa * fb >= 0:
                return {"error": "El método no es aplicable. La función no cambia de signo en [a, b]."}
        except Exception as e:
            return {"error": f"Error al evaluar función en los extremos: {str(e)}"}

        # Verificación de continuidad en el intervalo
        try:
            x_vals = np.linspace(a, b, 100)
            f_vals = f(x_vals)
            if np.any(np.isnan(f_vals)) or np.any(np.isinf(f_vals)):
                return {"error": "La función presenta discontinuidades o singularidades en [a, b]."}
        except Exception as e:
            return {"error": f"Error al verificar continuidad: {str(e)}"}

        # Inicialización de variables
        iteraciones = []
        c_vals = []
        convergencia = False
        c_ant = None

        # Iteración principal
        for itera in range(1, iteramax + 1):
            try:
                # Cálculo del punto c
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

                # Almacenar valores para gráfica
                c_vals.append(c)
                
                # Calcular error
                error = abs(c - c_ant) if c_ant is not None else None
                
                # Registrar iteración
                iteraciones.append({
                    "iter": itera,
                    "a": a, "b": b, "c": c,
                    "f_a": fa, "f_b": fb, "f_c": fc,
                    "error": error
                })

                # Verificar convergencia
                if error is not None and error < tol:
                    convergencia = True
                    break

                # Actualizar intervalo
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

        # Manejo de resultados según convergencia
        if not convergencia:
            return {
                "error": f"No se alcanzó la convergencia en {iteramax} iteraciones. Último error: {error:.6f}",
                "ultimo_valor": c,
                "iteraciones_realizadas": itera,
                "iteraciones": iteraciones
            }

        # Generar gráfica solo si convergió
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfica 1: Evolución de f(c)
        ax1.plot(c_vals, [f(ci) for ci in c_vals], 'ro-', label='f(c)')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_title("Evolución de f(c)")
        ax1.set_xlabel("c")
        ax1.set_ylabel("f(c)")
        ax1.grid(True)
        ax1.legend()
        
        # Gráfica 2: Función en el intervalo
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

        # Convertir a imagen base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        grafica_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": c,
            "iteraciones": iteraciones,
            "grafica_base64": grafica_base64,
            "convergencia": True,
            "iteraciones_totales": itera
        }

    except sp.SympifyError:
        return {"error": "Expresión matemática inválida. Use formato como 'cos(x) + x**2'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}