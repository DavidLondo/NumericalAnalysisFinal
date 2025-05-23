import sympy as sp
import numpy as np

def punto_fijo(g_input, x0, tol, iteramax=100):
    """
    Método de Punto Fijo mejorado
    
    Args:
        g_input: Función (str, sympy o lambda)
        x0: Punto inicial
        tol: Tolerancia
        iteramax: Máximo iteraciones (default 100)
    """
    x = sp.symbols('x')
    
    try:
        # 1. Conversión flexible de la función
        if isinstance(g_input, str):
            g_sym = sp.sympify(g_input)
        elif callable(g_input):
            g_sym = g_input(x) if hasattr(g_input, '__call__') else g_input
        else:
            g_sym = g_input
            
        # 2. Función numérica con soporte extendido
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
        
        # 3. Validación inicial
        try:
            test_val = g(x0)
            if np.isnan(test_val):
                return {"error": f"g(x) no está definida en x0 = {x0}"}
        except Exception as e:
            return {"error": f"Error al evaluar g(x0): {str(e)}"}

        iteraciones = []
        xi = x0
        
        for i in range(1, iteramax + 1):
            try:
                xi_new = g(xi)
                
                if np.isnan(xi_new):
                    return {"error": f"g(x) no definida en x = {xi:.6f}",
                            "iteraciones": iteraciones}
                
                error = abs(xi_new - xi)
                
                iteraciones.append({
                    "iter": i,
                    "xi": xi,
                    "f_xi": xi_new,
                    "error": error if i > 1 else None
                })
                
                if error < tol:
                    return {
                        "resultado": xi_new,
                        "iteraciones": iteraciones
                    }
                    
                xi = xi_new
                
            except Exception as e:
                return {"error": f"Error en iteración {i}: {str(e)}",
                        "iteraciones": iteraciones}
        
        return {
            "resultado": xi,
            "iteraciones": iteraciones,
            "warning": "Máximo de iteraciones alcanzado"
        }
        
    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'cos(x)', '0.5*(x + 2/x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}