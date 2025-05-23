from flask import Flask, render_template, request
from methods.biseccion import biseccion
from methods.reglafalsa import regla_falsa
from methods.puntofijo import punto_fijo
from methods.newton import newton_raphson
from methods.secante import secante
from methods.raicesmultiples import raices_multiples
import sympy as sp
import math

app = Flask(__name__)

@app.route("/inicio")
def inicio():
    return render_template("home.html")

@app.route("/capitulo-1", methods=["GET", "POST"])
def capitulo_1():
    def generar_informe_comparativo_raices(fx, gx=None, x0=None, x1=None, a=None, b=None, dfx=None, ddfx=None, tol=0.001):
        resultados = {}

        # Asignación inteligente de parámetros
        a_equiv = a if a is not None else x0
        b_equiv = b if b is not None else x1
        x0_equiv = x0 if x0 is not None else a
        x1_equiv = x1 if x1 is not None else b

        metodos = {
            "Bisección": lambda: biseccion(fx, a_equiv, b_equiv, tol) if a_equiv is not None and b_equiv is not None else {'error': 'Se requieren a y b (o x0 y x1)'},
            "Regla Falsa": lambda: regla_falsa(fx, a_equiv, b_equiv, tol) if a_equiv is not None and b_equiv is not None else {'error': 'Se requieren a y b (o x0 y x1)'},
            "Punto Fijo": lambda: punto_fijo(gx, x0_equiv, tol) if gx is not None and x0_equiv is not None else {'error': 'Se requieren g(x) y x0 (o a)'},
            "Newton-Raphson": lambda: newton_raphson(fx, x0_equiv, tol) if x0_equiv is not None else {'error': 'Se requiere x0 (o a)'},
            "Secante": lambda: secante(fx, x0_equiv, x1_equiv, tol) if x0_equiv is not None and x1_equiv is not None else {'error': 'Se requieren x0 y x1 (o a y b)'},
            "Raíces Múltiples": lambda: raices_multiples(fx, dfx, ddfx, x0_equiv, tol) if all([dfx, ddfx, x0_equiv is not None]) else {'error': 'Se requieren derivadas y x0 (o a)'}
        }

        for nombre, funcion in metodos.items():
            try:
                resultado = funcion()
                if 'error' in resultado:
                    resultados[nombre] = {'error': resultado['error']}
                else:
                    iteraciones = resultado.get('iteraciones', [])
                    n_iter = len(iteraciones)
                    error_final = iteraciones[-1]["error"] if iteraciones and "error" in iteraciones[-1] else None
                    
                    resultados[nombre] = {
                        'solucion': resultado['resultado'],
                        'iteraciones': n_iter,
                        'error_final': error_final
                    }
            except Exception as e:
                resultados[nombre] = {'error': str(e)}

        # Filtrar métodos que tuvieron éxito
        metodos_exitosos = {k: v for k, v in resultados.items() if 'error' not in v}
        
        if metodos_exitosos:
            # Ordenar primero por número de iteraciones, luego por error final
            mejor_metodo = min(
                metodos_exitosos.items(),
                key=lambda x: (x[1]['iteraciones'], x[1]['error_final'] if x[1]['error_final'] is not None else float('inf'))
            )
            mejor_nombre = mejor_metodo[0]
        else:
            mejor_nombre = "Ninguno"

        return resultados, mejor_nombre


    # Variables iniciales
    resultado_biseccion = error_biseccion = None
    resultado_reglafalsa = error_reglafalsa = None
    resultado_puntofijo = error_puntofijo = None
    resultado_newton = error_newton = None
    resultado_secante = error_secante = None
    resultado_raicesmultiples = error_raicesmultiples = None
    resultados_comparativos = mejor_metodo = None
    informe_comparativo = False
    metodo_actual = 'biseccion'

    if request.method == 'POST':
        metodo = request.form.get('metodo')
        metodo_actual = metodo
        informe_comparativo = 'informe_comparativo' in request.form

        # Inicialización de variables comunes
        fx = gx = dfx = ddfx = x0 = x1 = a = b = tol = None

        try:
            if metodo == 'biseccion':
                fx = request.form.get('f_biseccion')
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                tol = float(request.form.get('tol', 0.001))
                resultado = biseccion(fx, a, b, tol)
                resultado_biseccion = resultado if 'error' not in resultado else None
                error_biseccion = resultado.get('error')

            elif metodo == 'reglafalsa':
                fx = request.form.get('f_reglafalsa')
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                tol = float(request.form.get('tol', 0.001))
                resultado = regla_falsa(fx, a, b, tol)
                resultado_reglafalsa = resultado if 'error' not in resultado else None
                error_reglafalsa = resultado.get('error')

            elif metodo == 'puntofijo':
                gx = request.form.get('g_puntofijo')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = punto_fijo(gx, x0, tol)
                resultado_puntofijo = resultado if 'error' not in resultado else None
                error_puntofijo = resultado.get('error')

            elif metodo == 'newton':
                fx = request.form.get('f_newton')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = newton_raphson(fx, x0, tol)
                resultado_newton = resultado if 'error' not in resultado else None
                error_newton = resultado.get('error')

            elif metodo == 'secante':
                fx = request.form.get('f_secante')
                x0 = float(request.form.get('x0'))
                x1 = float(request.form.get('x1'))
                tol = float(request.form.get('tol', 0.001))
                resultado = secante(fx, x0, x1, tol)
                resultado_secante = resultado if 'error' not in resultado else None
                error_secante = resultado.get('error')

            elif metodo == 'raicesmultiples':
                fx = request.form.get('f_rm')
                dfx = request.form.get('df_rm')
                ddfx = request.form.get('ddf_rm')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = raices_multiples(fx, dfx, ddfx, x0, tol)
                resultado_raicesmultiples = resultado if 'error' not in resultado else None
                error_raicesmultiples = resultado.get('error')

            # ✅ Ejecutar informe comparativo si la checkbox está marcada
            if informe_comparativo:
                resultados_comparativos, mejor_metodo = generar_informe_comparativo_raices(
                    fx=fx, x0=x0, x1=x1, a=a, b=b, dfx=dfx, ddfx=ddfx, tol=tol
                )

        except Exception as e:
            msg = f"Error en los datos: {e}"
            if metodo == 'biseccion': error_biseccion = msg
            elif metodo == 'reglafalsa': error_reglafalsa = msg
            elif metodo == 'puntofijo': error_puntofijo = msg
            elif metodo == 'newton': error_newton = msg
            elif metodo == 'secante': error_secante = msg
            elif metodo == 'raicesmultiples': error_raicesmultiples = msg

    return render_template("chapter1.html",
                           resultado_biseccion=resultado_biseccion,
                           error_biseccion=error_biseccion,
                           resultado_reglafalsa=resultado_reglafalsa,
                           error_reglafalsa=error_reglafalsa,
                           resultado_puntofijo=resultado_puntofijo,
                           error_puntofijo=error_puntofijo,
                           resultado_newton=resultado_newton,
                           error_newton=error_newton,
                           resultado_secante=resultado_secante,
                           error_secante=error_secante,
                           resultado_raicesmultiples=resultado_raicesmultiples,
                           error_raicesmultiples=error_raicesmultiples,
                           resultados_comparativos=resultados_comparativos,
                           mejor_metodo=mejor_metodo,
                           informe_comparativo=informe_comparativo,
                           metodo_actual=metodo_actual)



@app.route("/capitulo-2")
def capitulo_2():
    return render_template("chapter2.html")

@app.route("/capitulo-3")
def capitulo_3():
    return render_template("chapter3.html")

if __name__ == "__main__":
    app.run(debug=True)