import sympy as sp
import numpy as np

def regla_falsa(fx_input, a, b, tol, iteramax=100):
    """
    Método de Regla Falsa mejorado
    
    Args:
        fx_input: Función (string, sympy o lambda)
        a: Extremo inferior del intervalo
        b: Extremo superior del intervalo
        tol: Tolerancia
        iteramax: Máximo de iteraciones (default 100)
    """
    x = sp.symbols('x')
    
    try:
        # 1. Conversión flexible de la función
        if isinstance(fx_input, str):
            # Reemplazar notación alternativa y convertir a sympy
            fx_str = fx_input.replace('^', '**').replace('sen', 'sin')
            f_expr = sp.sympify(fx_str)
        else:
            f_expr = fx_input
            
        # 2. Crear función numérica con soporte extendido
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
        
        # 3. Validación inicial
        try:
            fa, fb = f(a), f(b)
            if np.isnan(fa) or np.isnan(fb):
                return {"error": "La función no está definida en los extremos del intervalo"}
                
            if fa * fb >= 0:
                return {"error": "El método no es aplicable. La función no cambia de signo en el intervalo."}
        except Exception as e:
            return {"error": f"Error al evaluar la función: {str(e)}"}

        iteraciones = []
        c_ant = None
        
        for itera in range(1, iteramax + 1):
            try:
                # 4. Cálculo del punto c
                c = b - fb * (a - b) / (fa - fb)
                fc = f(c)
                
                if np.isnan(fc):
                    return {"error": f"La función no está definida en c = {c:.6f}",
                            "iteraciones": iteraciones}
                
                # 5. Cálculo del error
                error = abs(c - c_ant) if c_ant is not None else None
                
                iteraciones.append({
                    "iter": itera,
                    "a": a,
                    "b": b,
                    "c": c,
                    "f_a": fa,
                    "f_b": fb,
                    "f_c": fc,
                    "error": error
                })
                
                # 6. Verificación de convergencia
                if error is not None and error < tol:
                    break
                    
                # 7. Actualización del intervalo
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    
                c_ant = c
                
            except ZeroDivisionError:
                return {"error": "División por cero. La función puede ser constante.",
                        "iteraciones": iteraciones}
            except Exception as e:
                return {"error": f"Error en iteración {itera}: {str(e)}",
                        "iteraciones": iteraciones}
        
        return {
            "resultado": c,
            "iteraciones": iteraciones
        }
        
    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'cos(x)-x', 'exp(-x)-log(x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}