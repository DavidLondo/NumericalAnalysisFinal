import sympy as sp
import numpy as np
from sympy import sympify

def biseccion(fx, a, b, tol):
    """
    Método de bisección adaptado para Flask.
    
    Args:
        fx: Función como string desde el formulario (ej: "cos(x) + x**2")
        a: Límite inferior del intervalo
        b: Límite superior del intervalo
        tol: Tolerancia para el criterio de parada
    """
    x = sp.symbols('x')
    
    try:
        # Convertir string a expresión simbólica
        f_sym = sp.sympify(fx)
        
        # Crear función numérica con soporte para funciones especiales
        f = sp.lambdify(x, f_sym, modules=[
            'numpy',
            {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
             'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
             'ln': np.log, 'e': np.e, 'pi': np.pi}
        ])
        
        resultados = []
        
        # Verificación inicial del intervalo
        try:
            fa, fb = f(a), f(b)
            if np.isnan(fa) or np.isnan(fb):
                return {"error": "La función no está definida en los extremos del intervalo"}
                
            if fa * fb >= 0:
                return {"error": "El método no es aplicable. La función no cambia de signo en el intervalo."}
        except Exception as e:
            return {"error": f"Error al evaluar la función: {str(e)}"}

        i = 1
        error = None
        x_ant = None

        while abs(b - a) > tol:
            try:
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

                if f_xi == 0:
                    break
                elif f(a) * f_xi < 0:
                    b = xi
                else:
                    a = xi

                x_ant = xi
                i += 1
                
            except Exception as e:
                return {"error": f"Error en iteración {i}: {str(e)}", "iteraciones": resultados}

        return {
            "resultado": xi,
            "iteraciones": resultados
        }
        
    except sp.SympifyError:
        return {"error": "Expresión matemática inválida. Use formato como 'cos(x) + x**2'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}