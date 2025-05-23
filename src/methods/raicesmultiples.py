import math

def raices_multiples(Fun, derivada_fx1, derivada_fx2, X0, tol, Niter=100):
    """
    Método de Raíces Múltiples modificado para aceptar funciones complejas
    pero manteniendo la estructura original.
    """
    # Creamos un entorno seguro para eval()
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'pow': pow, '**': pow
    }
    
    # Preparamos las funciones para evaluación
    def evaluar(expr, x_val):
        try:
            safe_dict['x'] = x_val
            return eval(expr, {"__builtins__": None}, safe_dict)
        except:
            raise ValueError(f"No se pudo evaluar la expresión: {expr}")

    x = X0
    iteraciones = []
    c = 0
    
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
            try:
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
                
                # Actualizamos valores
                x = x_nuevo
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
                    
            except Exception as e:
                return {
                    "resultado": x,
                    "iteraciones": iteraciones,
                    "error": f"Error en iteración {c}: {str(e)}"
                }

        return {
            "resultado": x,
            "iteraciones": iteraciones
        }
        
    except Exception as e:
        return {
            "error": f"Error inicial: {str(e)}"
        }