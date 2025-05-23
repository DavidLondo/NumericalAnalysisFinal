import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def secante(fx_input, X0, X1, tol, iteramax=100):
    """
    Método de la Secante mejorado con manejo robusto de convergencia
    
    Args:
        fx_input: Función como string o expresión SymPy
        X0: Primer punto inicial
        X1: Segundo punto inicial
        tol: Tolerancia para el criterio de parada
        iteramax: Máximo número de iteraciones permitidas
    
    Returns:
        Diccionario con:
        - resultado: aproximación final (si converge)
        - iteraciones: tabla de iteraciones
        - grafica_base64: gráfica en base64 (si converge)
        - error: mensaje si no converge o hay error
        - convergencia: booleano indicando si convergió
        - ultimo_valor: última aproximación calculada
        - advertencia: mensajes no críticos
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
        
        # Validación inicial de puntos
        try:
            f0, f1 = f(X0), f(X1)
            if np.isnan(f0) or np.isnan(f1):
                return {
                    "error": "La función no está definida en los puntos iniciales.",
                    "convergencia": False,
                    "ultimo_valor": None,
                    "iteraciones": []
                }
        except Exception as e:
            return {
                "error": f"Error al evaluar función en puntos iniciales: {str(e)}",
                "convergencia": False,
                "ultimo_valor": None,
                "iteraciones": []
            }

        # Inicialización de variables
        xn = [float(X0), float(X1)]
        fn = [f0, f1]
        iteraciones = []
        convergencia = False
        advertencia = None

        # Primera iteración (solo para registro)
        iteraciones.append({
            "iter": 0,
            "Xi": xn[1],
            "f_Xi": fn[1],
            "error": None
        })

        # Iteración principal
        for c in range(1, iteramax + 1):
            try:
                denominator = (fn[-1] - fn[-2])
                
                # Verificar divisor muy pequeño para evitar división por cero
                if abs(denominator) < 1e-15:
                    advertencia = "Divisor cercano a cero detectado. Posible convergencia lenta o error."
                    break
                
                # Calcular nueva aproximación
                x_new = xn[-1] - fn[-1] * (xn[-1] - xn[-2]) / denominator
                f_new = f(x_new)

                # Verificar valores no definidos o infinitos
                if np.isnan(f_new) or np.isinf(f_new):
                    return {
                        "error": f"La función no está definida en x = {x_new:.6f}.",
                        "iteraciones": iteraciones,
                        "ultimo_valor": xn[-1],
                        "convergencia": False,
                        "advertencia": advertencia
                    }

                # Almacenar valores
                xn.append(x_new)
                fn.append(f_new)
                error = abs(xn[-1] - xn[-2])
                
                # Registrar iteración
                iteraciones.append({
                    "iter": c,
                    "Xi": xn[-1],
                    "f_Xi": fn[-1],
                    "error": error
                })

                # Verificar convergencia
                if error < tol or abs(f_new) < 1e-12:
                    convergencia = True
                    break
                    
            except Exception as e:
                return {
                    "error": f"Error en iteración {c}: {str(e)}",
                    "iteraciones": iteraciones,
                    "ultimo_valor": xn[-1],
                    "convergencia": False,
                    "advertencia": advertencia
                }

        # Manejo de resultados según convergencia
        if not convergencia:
            return {
                "error": f"No se alcanzó la convergencia en {iteramax} iteraciones. Último error: {error:.6e}" if 'error' in locals() else "No se alcanzó la convergencia.",
                "ultimo_valor": xn[-1],
                "iteraciones_realizadas": c,
                "iteraciones": iteraciones,
                "convergencia": False,
                "advertencia": advertencia
            }

        # ----------- GRÁFICA (solo si convergió) -----------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfica 1: Evolución de f(Xn)
        ax1.plot(xn[2:], fn[2:], 'bo-', label='f(Xn)')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_title("Evolución de f(Xn)")
        ax1.set_xlabel("Xn")
        ax1.set_ylabel("f(Xn)")
        ax1.grid(True)
        ax1.legend()
        
        # Gráfica 2: Convergencia del error
        errores = [it['error'] for it in iteraciones if it['error'] is not None]
        if errores:
            ax2.semilogy(range(1, len(errores)+1), errores, 'g-', marker='o')
            ax2.set_title('Convergencia del Error')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Error (escala log)')
            ax2.grid(True)
        
        plt.tight_layout()

        # Convertir a imagen base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        grafica_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "resultado": xn[-1],
            "iteraciones": iteraciones,
            "grafica_base64": grafica_base64,
            "convergencia": True,
            "iteraciones_totales": c,
            "advertencia": advertencia
        }

    except sp.SympifyError:
        return {
            "error": "Expresión no válida. Ejemplos: 'x*tan(x)-1', 'exp(-x)-cos(x)'",
            "convergencia": False,
            "ultimo_valor": None,
            "iteraciones": []
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "convergencia": False,
            "ultimo_valor": None,
            "iteraciones": []
        }
