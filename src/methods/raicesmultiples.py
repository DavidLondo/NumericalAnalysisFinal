import math
import matplotlib.pyplot as plt
import io
import base64

def raices_multiples(Fun, derivada_fx1, derivada_fx2, X0, tol, Niter=100):
    """
    Método de Raíces Múltiples mejorado con manejo robusto de convergencia
    
    Args:
        Fun: Función como string
        derivada_fx1: Primera derivada de la función (string)
        derivada_fx2: Segunda derivada de la función (string)
        X0: Punto inicial (float)
        tol: Tolerancia para el criterio de parada (float)
        Niter: Máximo número de iteraciones permitidas (int)
    
    Returns:
        Diccionario con:
        - resultado: aproximación final (si converge)
        - iteraciones: tabla de iteraciones
        - grafica_base64: gráfica en base64 (si converge)
        - error: mensaje si no converge o hay error
        - convergencia: booleano indicando si convergió
        - ultimo_valor: última aproximación calculada
        - advertencia: posibles advertencias (denominador cercano a cero)
    """
    # Diccionario seguro para evaluación de expresiones
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'pow': pow
    }
    
    def evaluar(expr, x_val):
        """Evalúa una expresión matemática con seguridad"""
        try:
            safe_dict['x'] = x_val
            return eval(expr, {"__builtins__": None}, safe_dict)
        except Exception as e:
            raise ValueError(f"Error al evaluar '{expr}' en x={x_val}: {str(e)}")

    iteraciones = []
    x = X0
    convergencia = False
    error = None
    advertencia = None
    x_vals = [x]

    try:
        # Evaluar valores iniciales para validar expresiones y denominador
        f = evaluar(Fun, x)
        derivada1 = evaluar(derivada_fx1, x)
        derivada2 = evaluar(derivada_fx2, x)
        denominador = derivada1**2 - (f * derivada2)
        
        # Registrar iteración inicial
        iteraciones.append({
            "iter": 0,
            "Xi": x,
            "f_Xi": f,
            "derivada1": derivada1,
            "derivada2": derivada2,
            "error": None
        })

        for c in range(1, Niter + 1):
            try:
                if abs(denominador) < 1e-15:
                    advertencia = "Denominador cercano a cero. Método puede ser inestable."
                    break

                x_nuevo = x - (f * derivada1) / denominador
                error = abs(x_nuevo - x)
                
                x = x_nuevo
                x_vals.append(x)

                f = evaluar(Fun, x)
                derivada1 = evaluar(derivada_fx1, x)
                derivada2 = evaluar(derivada_fx2, x)
                denominador = derivada1**2 - (f * derivada2)

                iteraciones.append({
                    "iter": c,
                    "Xi": x,
                    "f_Xi": f,
                    "error": error
                })

                if error < tol or abs(f) < 1e-12:
                    convergencia = True
                    break

            except Exception as e:
                return {
                    "error": f"Error en iteración {c}: {str(e)}",
                    "iteraciones": iteraciones,
                    "ultimo_valor": x,
                    "convergencia": False
                }

        if not convergencia:
            return {
                "error": f"No se alcanzó la convergencia en {Niter} iteraciones. Último error: {error:.6f}",
                "ultimo_valor": x,
                "iteraciones_realizadas": c,
                "iteraciones": iteraciones,
                "convergencia": False,
                "advertencia": advertencia
            }

        # Gráficas solo si hubo convergencia
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        y_vals = [evaluar(Fun, xi) for xi in x_vals]
        ax1.plot(x_vals, y_vals, 'bo-', label='f(xi)')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_title("Evolución de f(xi)")
        ax1.set_xlabel("xi")
        ax1.set_ylabel("f(xi)")
        ax1.grid(True)
        ax1.legend()

        errores = [it['error'] for it in iteraciones if it['error'] is not None]
        if errores:
            ax2.semilogy(range(1, len(errores) + 1), errores, 'g-', marker='o')
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
            "resultado": x,
            "iteraciones": iteraciones,
            "grafica_base64": image_base64,
            "convergencia": True,
            "iteraciones_totales": c,
            "advertencia": advertencia
        }

    except Exception as e:
        return {
            "error": f"Error inicial: {str(e)}",
            "convergencia": False
        }
