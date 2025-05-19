from flask import Flask, render_template, request
from methods.biseccion import biseccion
from methods.reglafalsa import regla_falsa
import sympy as sp

app = Flask(__name__)

@app.route("/inicio")
def inicio():
    return render_template("home.html")

@app.route("/capitulo-1", methods=["GET", "POST"])
def capitulo_1():
    # Variables de resultado y error por método
    resultado_biseccion = error_biseccion = None
    resultado_reglafalsa = error_reglafalsa = None
    resultado_puntofijo = error_puntofijo = None
    resultado_newton = error_newton = None
    resultado_secante = error_secante = None
    resultado_raicesmultiples = error_raicesmultiples = None
    metodo_actual = 'biseccion'

    if request.method == 'POST':
        metodo = request.form.get('metodo')
        metodo_actual = metodo

        try:
            if metodo == 'biseccion':
                fx = request.form.get('f_biseccion')
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                tol = float(request.form.get('tol', 0.001))
                resultado = biseccion(fx, a, b, tol)

                if 'error' in resultado:
                    error_biseccion = resultado['error']
                else:
                    resultado_biseccion = resultado
                pass

            elif metodo == 'reglafalsa':
                fx = request.form.get('f_reglafalsa')
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                tol = float(request.form.get('tol', 0.001))
                resultado = regla_falsa(fx, a, b, tol)

                if 'error' in resultado:
                    error_reglafalsa = resultado['error']
                else:
                    resultado_reglafalsa = resultado
                pass

            elif metodo == 'puntofijo':
                gx = request.form.get('g_puntofijo')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = punto_fijo(gx, x0, tol)

                if 'error' in resultado:
                    error_puntofijo = resultado['error']
                else:
                    resultado_puntofijo = resultado

            elif metodo == 'newton':
                fx = request.form.get('f_newton')
                dfx = request.form.get('df')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = newton_raphson(fx, dfx, x0, tol)

                if 'error' in resultado:
                    error_newton = resultado['error']
                else:
                    resultado_newton = resultado

            elif metodo == 'secante':
                fx = request.form.get('f_secante')
                x0 = float(request.form.get('x0'))
                x1 = float(request.form.get('x1'))
                tol = float(request.form.get('tol', 0.001))
                resultado = secante(fx, x0, x1, tol)

                if 'error' in resultado:
                    error_secante = resultado['error']
                else:
                    resultado_secante = resultado

            elif metodo == 'raicesmultiples':
                fx = request.form.get('f_rm')
                dfx = request.form.get('df_rm')
                ddfx = request.form.get('ddf_rm')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = raices_multiples(fx, dfx, ddfx, x0, tol)

                if 'error' in resultado:
                    error_raicesmultiples = resultado['error']
                else:
                    resultado_raicesmultiples = resultado

        except Exception as e:
            error_msg = f"Error en los datos: {e}"
            if metodo == 'biseccion':
                error_biseccion = error_msg
            elif metodo == 'reglafalsa':
                error_reglafalsa = error_msg
            elif metodo == 'puntofijo':
                error_puntofijo = error_msg
            elif metodo == 'newton':
                error_newton = error_msg
            elif metodo == 'secante':
                error_secante = error_msg
            elif metodo == 'raicesmultiples':
                error_raicesmultiples = error_msg

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
                           metodo_actual=metodo_actual)


@app.route("/capitulo-2")
def capitulo_2():
    return render_template("chapter2.html")

@app.route("/capitulo-3")
def capitulo_3():
    return render_template("chapter3.html")

if __name__ == "__main__":
    app.run(debug=True)