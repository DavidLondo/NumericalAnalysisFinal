import sympy as sp
import numpy as np

def secante(fx_input, X0, X1, tol, iteramax=100):
    """
    Método de la Secante mejorado
    
    Args:
        fx_input: Función (string, expresión sympy o lambda)
        X0: Primer punto inicial
        X1: Segundo punto inicial
        tol: Tolerancia
        iteramax: Máximo de iteraciones (default 100)
    """
    x = sp.symbols('x')
    
    try:
        # 1. Conversión flexible de la función
        if isinstance(fx_input, str):
            # Aceptar múltiples formatos (sen/sin, ^/**)
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
            f0, f1 = f(X0), f(X1)
            if np.isnan(f0) or np.isnan(f1):
                return {"error": "La función no está definida en los puntos iniciales"}
        except Exception as e:
            return {"error": f"Error al evaluar función en puntos iniciales: {str(e)}"}

        xn = [float(X0), float(X1)]  # Aseguramos tipo numérico
        fn = [f0, f1]
        iteraciones = []
        
        # Registro de iteración inicial
        iteraciones.append({
            "iter": 0,
            "Xi": xn[1],
            "f_Xi": fn[1],
            "error": None
        })

        for c in range(1, iteramax + 1):
            try:
                # 4. Cálculo del nuevo punto
                denominator = (fn[-1] - fn[-2])
                if abs(denominator) < 1e-15:
                    return {
                        "resultado": xn[-1],
                        "iteraciones": iteraciones,
                        "warning": "Divisor cercano a cero. Posible convergencia lenta."
                    }
                
                x_new = xn[-1] - fn[-1] * (xn[-1] - xn[-2]) / denominator
                f_new = f(x_new)
                
                if np.isnan(f_new):
                    return {
                        "resultado": xn[-1],
                        "iteraciones": iteraciones,
                        "error": f"Función no definida en x = {x_new:.6f}"
                    }
                
                # 5. Actualización de valores
                xn.append(x_new)
                fn.append(f_new)
                error = abs(xn[-1] - xn[-2])
                
                iteraciones.append({
                    "iter": c,
                    "Xi": xn[-1],
                    "f_Xi": fn[-1],
                    "error": error
                })
                
                # 6. Criterios de parada
                if error < tol or abs(f_new) < 1e-12:
                    break
                    
            except Exception as e:
                return {
                    "resultado": xn[-1],
                    "iteraciones": iteraciones,
                    "error": f"Error en iteración {c}: {str(e)}"
                }
        
        return {
            "resultado": xn[-1],
            "iteraciones": iteraciones
        }
        
    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'x*tan(x)-1', 'exp(-x)-cos(x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}