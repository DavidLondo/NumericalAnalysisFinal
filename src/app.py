from flask import Flask, render_template, request, redirect, url_for
from methods import *
import sympy as sp
import math

app = Flask(__name__)

# If the user goes to /, redirect to /inicio
@app.route("/")
def index():
    return redirect(url_for("inicio"))

@app.route("/inicio")
def inicio():
    return render_template("home.html")

@app.route("/capitulo-1", methods=["GET", "POST"])
def capitulo_1():
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
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                resultado = newton_raphson(fx, x0, tol)

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


@app.route("/capitulo-2", methods=["GET", "POST"])
def capitulo_2():
    def generar_informe_comparativo(A, b, x0, tol, w=1.1):
        resultados = {}
        metodos = {
            "Jacobi": jacobi,
            "Gauss-Seidel": gauss_seidel,
            "SOR": lambda A, b, x0, tol: sor(A, b, w, x0, tol)
        }

        for nombre, funcion in metodos.items():
            try:
                resultado = funcion(A, b, x0, tol)
                n_iter = len(resultado['iteraciones'])
                error_final = resultado['iteraciones'][-1][2]
                resultados[nombre] = {
                    'solucion': resultado['solucion'],
                    'iteraciones': n_iter,
                    'error_final': error_final
                }
            except Exception as e:
                resultados[nombre] = {'error': str(e)}

        mejor = min(
            (m for m in resultados.items() if 'error' not in m[1]),
            key=lambda x: (x[1]['iteraciones'], x[1]['error_final']),
            default=None
        )
        print("RESULTAOS:", resultados)
        return resultados, mejor[0] if mejor else "Ninguno"
    
    resultado_jacobi = resultado_gauss = resultado_sor = None
    error_jacobi = error_gauss = error_sor = None
    radio_espectral = puede_converger = None
    informe = mejor_metodo = None
    metodo_actual = "jacobi"

    if request.method == "POST":
        metodo = request.form.get("metodo")
        metodo_actual = metodo
        try:
            A = eval(request.form.get("matriz"))
            b = eval(request.form.get("vector_b"))
            x0 = eval(request.form.get("x0")) if request.form.get("x0") else None
            tol = float(request.form.get("tol", 0.001))

            if metodo == "jacobi":
                resultado_jacobi = jacobi(A, b, x0, tol)
                if A:
                    radio_espectral, puede_converger = analizar_convergencia(A, metodo)
                if request.form.get("generar_informe"):
                    informe, mejor_metodo = generar_informe_comparativo(A, b, x0, tol)
                    print("informe:", informe)
                    print("mejor_metodo:", mejor_metodo)
                print("resultado_jacobi:", resultado_jacobi)
            elif metodo == "gaussseidel":
                resultado_gauss = gauss_seidel(A, b, x0, tol)
                if A:
                    radio_espectral, puede_converger = analizar_convergencia(A, metodo)
                if request.form.get("generar_informe"):
                    informe, mejor_metodo = generar_informe_comparativo(A, b, x0, tol)
                    print("informe:", informe)
                    print("mejor_metodo:", mejor_metodo)
                print("resultado_gauss:", resultado_gauss)
            elif metodo == "sor":
                w = float(request.form.get("w", 1.1))
                resultado_sor = sor(A, b, w, x0, tol)
                if A:
                    radio_espectral, puede_converger = analizar_convergencia(A, metodo, w)
                if request.form.get("generar_informe"):
                    informe, mejor_metodo = generar_informe_comparativo(A, b, x0, tol, w)
                    print("informe:", informe)
                    print("mejor_metodo:", mejor_metodo)
                print("resultado_sor:", resultado_sor)

        except Exception as e:
            if metodo == "jacobi":
                error_jacobi = str(e)
            elif metodo == "gaussseidel":
                error_gauss = str(e)
            elif metodo == "sor":
                error_sor = str(e)
    
    print("metodo_actual:"+ metodo_actual+"|")

    return render_template("chapter2.html",
                            resultado_jacobi=resultado_jacobi,
                            error_jacobi=error_jacobi,
                            resultado_gauss=resultado_gauss,
                            error_gauss=error_gauss,
                            resultado_sor=resultado_sor,
                            error_sor=error_sor,
                            metodo_actual=metodo_actual,
                            radio_espectral=radio_espectral,
                            puede_converger=puede_converger,
                            informe=informe,
                            mejor_metodo=mejor_metodo)

@app.route("/capitulo-3")
def capitulo_3():
    return render_template("chapter3.html")

if __name__ == "__main__":
    app.run(debug=True)