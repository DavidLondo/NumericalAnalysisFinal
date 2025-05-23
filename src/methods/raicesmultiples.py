import math
import matplotlib.pyplot as plt
import io
import base64

def raices_multiples(Fun, derivada_fx1, derivada_fx2, X0, tol, Niter=100):
    """
    Método de Raíces Múltiples mejorado con generación de gráfica (base64)
    """
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'pow': pow, '**': pow
    }
    
    def evaluar(expr, x_val):
        try:
            safe_dict['x'] = x_val
            return eval(expr, {"__builtins__": None}, safe_dict)
        except:
            raise ValueError(f"No se pudo evaluar la expresión: {expr}")

    x = X0
    iteraciones = []
    c = 0
    x_vals = [x]

    try:
        f = evaluar(Fun, x)
        derivada1 = evaluar(derivada_fx1, x)
        derivada2 = evaluar(derivada_fx2, x)
        denominador = derivada1**2 - (f * derivada2)
        
        iteraciones.append({
            "iter": c,
            "Xi": x,
            "f_Xi": f,
            "error": None
        })

        while c < Niter:
            if abs(f) < tol:
                break

            if denominador == 0:
                return {
                    "resultado": x,
                    "iteraciones": iteraciones,
                    "error": "Denominador cero. El método falló."
                }

            x_nuevo = x - (f * derivada1) / denominador
            Error = abs(x_nuevo - x)
            
            x = x_nuevo
            x_vals.append(x)

            f = evaluar(Fun, x)
            derivada1 = evaluar(derivada_fx1, x)
            derivada2 = evaluar(derivada_fx2, x)
            denominador = derivada1**2 - (f * derivada2)
            
            c += 1
            iteraciones.append({
                "iter": c,
                "Xi": x,
                "f_Xi": f,
                "error": Error
            })

            if Error < tol:
                break

        # ----------- GRÁFICA -----------
        fig, ax = plt.subplots()
        y_vals = [evaluar(Fun, xi) for xi in x_vals]

        ax.plot(x_vals, y_vals, 'bo-', label='f(xi)')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title("Raíces Múltiples - f(x) en iteraciones")
        ax.set_xlabel("xi")
        ax.set_ylabel("f(xi)")
        ax.grid(True)
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": x,
            "iteraciones": iteraciones,
            "grafica_base64": image_base64
        }

    except Exception as e:
        return {
            "error": f"Error inicial: {str(e)}"
        }
