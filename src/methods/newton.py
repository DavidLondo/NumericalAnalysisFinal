import sympy as sp
import numpy as np

def newton_raphson(fx_str, x0, tol, iteramax=100):
    """
    Método de Newton-Raphson mejorado para funciones complejas
    
    Args:
        fx_str: Función como string (ej: "cos(x)*exp(-x)")
        x0: Punto inicial
        tol: Tolerancia
        iteramax: Máximo de iteraciones (default 100)
    """
    x = sp.symbols('x')
    
    try:
        # 1. Conversión a expresión simbólica
        fx = sp.sympify(fx_str)
        
        # 2. Cálculo automático de derivada
        dfx = sp.diff(fx, x)
        
        # 3. Funciones numéricas con soporte extendido
        f = sp.lambdify(x, fx, modules=[
            'numpy',
            {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'ln': np.log, 'e': np.e, 'pi': np.pi,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'sec': lambda x: 1/np.cos(x), 'csc': lambda x: 1/np.sin(x)
            }
        ])
        
        df = sp.lambdify(x, dfx, modules='numpy')
        
        # 4. Validación inicial
        try:
            test_val = f(x0)
            if np.isnan(test_val):
                return {"error": f"La función no está definida en x0 = {x0}"}
        except Exception as e:
            return {"error": f"Error al evaluar función en x0: {str(e)}"}

        iteraciones = []
        xi = x0
        
        for i in range(1, iteramax + 1):
            try:
                fxi = f(xi)
                dfxi = df(xi)
                
                if np.isnan(fxi) or np.isnan(dfxi):
                    return {"error": f"Función o derivada no definida en iteración {i}, x = {xi:.6f}",
                            "iteraciones": iteraciones}
                
                if dfxi == 0:
                    return {"error": f"Derivada cero en iteración {i}. Método falló.",
                            "iteraciones": iteraciones}
                
                xi_new = xi - fxi/dfxi
                error = abs(xi_new - xi)
                
                iteraciones.append({
                    "iter": i,
                    "xi": xi,
                    "f_xi": fxi,
                    "df_xi": dfxi,
                    "error": error if i > 1 else None
                })
                
                if error < tol:
                    break
                    
                xi = xi_new
                
            except Exception as e:
                return {"error": f"Error en iteración {i}: {str(e)}",
                        "iteraciones": iteraciones}
        
        return {
            "resultado": xi,
            "iteraciones": iteraciones
        }
        
    except sp.SympifyError:
        return {"error": "Expresión no válida. Ejemplos: 'x*tan(x)-1', 'exp(-x)-cos(x)'"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}