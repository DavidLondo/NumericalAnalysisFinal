import math
import matplotlib.pyplot as plt
import io
import base64

def raices_multiples(Fun, derivada_fx1, derivada_fx2, X0, tol, Niter=100):
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'pow': pow
    }
    
    def evaluar(expr, x_val):
        safe_dict['x'] = x_val
        return eval(expr, {"__builtins__": None}, safe_dict)

    iteraciones = []
    x = X0
    convergencia = False
    advertencia = None
    x_vals = [x]

    try:
        f = evaluar(Fun, x)
        derivada1 = evaluar(derivada_fx1, x)
        derivada2 = evaluar(derivada_fx2, x)
        denominador = derivada1**2 - (f * derivada2)
        
        iteraciones.append({
            "iter": 0,
            "Xi": x,
            "f_Xi": f,
            "error": None
        })

        for c in range(1, Niter + 1):
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

        if not convergencia:
            return {
                "error": f"No se alcanzó la convergencia en {Niter} iteraciones. Último error: {error:.6f}",
                "iteraciones": iteraciones,
                "convergencia": False,
                "advertencia": advertencia
            }

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
            "advertencia": advertencia
        }

    except Exception as e:
        return {
            "error": f"Error inicial: {str(e)}",
            "convergencia": False
        }
