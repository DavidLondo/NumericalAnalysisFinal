from flask import Flask, render_template, request, redirect, url_for
from methods import *
import sympy as sp
import math
import numpy as np

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
    def generar_informe_comparativo_raices(fx, gx=None, x0=None, x1=None, a=None, b=None, dfx=None, ddfx=None, tol=0.001):
        resultados = {}

        # Asignación inteligente de parámetros con valor por defecto para x1/b
        a_equiv = a if a is not None else x0
        b_equiv = b if b is not None else x1
        
        # Si no se proporciona x1 o b, se establece como x0/a + 1
        if b_equiv is None and a_equiv is not None:
            b_equiv = a_equiv + 1
        
        x0_equiv = x0 if x0 is not None else a
        x1_equiv = x1 if x1 is not None else b
        
        # Si no se proporciona x1 o b, se establece como x0/a + 1
        if x1_equiv is None and x0_equiv is not None:
            x1_equiv = x0_equiv + 1

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
                itera = int(request.form.get('iter', 100))
                resultado = biseccion(fx, a, b, tol, itera)
                resultado_biseccion = resultado if 'error' not in resultado else None
                error_biseccion = resultado.get('error')

            elif metodo == 'reglafalsa':
                fx = request.form.get('f_reglafalsa')
                a = float(request.form.get('a'))
                b = float(request.form.get('b'))
                tol = float(request.form.get('tol', 0.001))
                itera = int(request.form.get('iter', 100))
                resultado = regla_falsa(fx, a, b, tol, itera)
                resultado_reglafalsa = resultado if 'error' not in resultado else None
                error_reglafalsa = resultado.get('error')

            elif metodo == 'puntofijo':
                gx = request.form.get('g_puntofijo')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                itera = int(request.form.get('iter', 100))
                resultado = punto_fijo(gx, x0, tol, itera)
                resultado_puntofijo = resultado if 'error' not in resultado else None
                error_puntofijo = resultado.get('error')

            elif metodo == 'newton':
                fx = request.form.get('f_newton')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                itera = int(request.form.get('iter', 100))
                resultado = newton_raphson(fx, x0, tol, itera)
                resultado_newton = resultado if 'error' not in resultado else None
                error_newton = resultado.get('error')

            elif metodo == 'secante':
                fx = request.form.get('f_secante')
                x0 = float(request.form.get('x0'))
                x1 = float(request.form.get('x1'))
                tol = float(request.form.get('tol', 0.001))
                itera = int(request.form.get('iter', 100))
                resultado = secante(fx, x0, x1, tol, itera)
                resultado_secante = resultado if 'error' not in resultado else None
                error_secante = resultado.get('error')

            elif metodo == 'raicesmultiples':
                fx = request.form.get('f_rm')
                dfx = request.form.get('df_rm')
                ddfx = request.form.get('ddf_rm')
                x0 = float(request.form.get('x0'))
                tol = float(request.form.get('tol', 0.001))
                itera = int(request.form.get('iter', 100))
                resultado = raices_multiples(fx, dfx, ddfx, x0, tol, itera)
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
            elif metodo == "gaussseidel":
                resultado_gauss = gauss_seidel(A, b, x0, tol)
                if A:
                    radio_espectral, puede_converger = analizar_convergencia(A, metodo)
                if request.form.get("generar_informe"):
                    informe, mejor_metodo = generar_informe_comparativo(A, b, x0, tol)
            elif metodo == "sor":
                w = float(request.form.get("w", 1.1))
                resultado_sor = sor(A, b, w, x0, tol)
                if A:
                    radio_espectral, puede_converger = analizar_convergencia(A, metodo, w)
                if request.form.get("generar_informe"):
                    informe, mejor_metodo = generar_informe_comparativo(A, b, x0, tol, w)

        except Exception as e:
            if metodo == "jacobi":
                error_jacobi = str(e)
            elif metodo == "gaussseidel":
                error_gauss = str(e)
            elif metodo == "sor":
                error_sor = str(e)

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

@app.route("/capitulo-3", methods=["GET", "POST"])
def capitulo_3():
    def pol_to_str(coef):
        grado = len(coef)
        # Transform coef to string eg. a*x^3 + b*x^2 + c*x + d
        str_coef = [f"{coef[i]}*x^{grado - i - 1}" for i in range(grado) if coef[i] != 0]
        str_coef = [s.replace("*x^1", "*x") for s in str_coef]
        str_coef = [s.replace("*x^0", "") for s in str_coef]
        for i in range(len(str_coef) - 1):
            if not str_coef[i + 1].startswith("-"):
                str_coef[i] += " +"
        return " ".join(str_coef)

    def generar_informe_comparativo(x, y, x_real, y_real):
        metodos = {
            "Vandermonde": vandermonde,
            "Newton": newtoninter,
            "Lagrange": lagrange
        }

        informe = {}
        errores = {}

        for nombre, funcion in metodos.items():
            try:
                resultado = funcion(x, y)
                coef = resultado['solucion']
                y_estimado = np.polyval(coef, x_real)
                error = abs(y_real - y_estimado)
                informe[nombre] = {
                    "solucion": pol_to_str(resultado['solucion']),
                    "error_validacion": error,
                    "y_estimado": y_estimado
                }
                errores[nombre] = error
            except Exception as e:
                informe[nombre] = {"error": str(e)}
                errores[nombre] = float('inf')

        # ----- Spline Lineal -----
        try:
            spline_lineal = spline_interpolation(x, y, grado=1)
            if "error" in spline_lineal:
                raise Exception(spline_lineal["error"])
            
            coef_splines = spline_lineal["solucion"]
            y_val = None

            for i in range(len(x) - 1):
                if x[i] <= x_real <= x[i + 1]:
                    coef = coef_splines[i]
                    d = len(coef) - 1
                    y_val = sum(coef[j] * x_real**(d - j) for j in range(d + 1))
                    break
            if y_val is None:
                y_val = 0

            error = abs(y_real - y_val)

            informe["Spline Lineal"] = {
                "coeficientes_tramos": [pol_to_str(i) for i in coef_splines],
                "error_validacion": error,
                "y_estimado": y_val
            }
            errores["Spline Lineal"] = error

        except Exception as e:
            informe["Spline Lineal"] = {"error": str(e)}
            errores["Spline Lineal"] = float('inf')

        # ----- Spline Cúbico -----
        try:
            spline_cubico = spline_interpolation(x, y, grado=3)
            if "error" in spline_cubico:
                raise Exception(spline_cubico["error"])
            
            coef_splines = spline_cubico["solucion"]
            y_val = None

            for i in range(len(x) - 1):
                if x[i] <= x_real <= x[i + 1]:
                    coef = coef_splines[i]
                    d = len(coef) - 1
                    y_val = sum(coef[j] * x_real**(d - j) for j in range(d + 1))
                    break
            if y_val is None:
                y_val = 0

            error = abs(y_real - y_val)

            informe["Spline Cúbico"] = {
                "coeficientes_tramos": [pol_to_str(i) for i in coef_splines],
                "error_validacion": error,
                "y_estimado": y_val
            }
            errores["Spline Cúbico"] = error

        except Exception as e:
            informe["Spline Cúbico"] = {"error": str(e)}
            errores["Spline Cúbico"] = float('inf')

        mejor = min(errores.items(), key=lambda x: x[1], default=("Ninguno", None))[0]
        return informe, mejor



    resultado_vandermonde = resultado_newton = resultado_lagrange = resultado_spline = None
    error_vandermonde = error_newton = error_lagrange = error_spline = None
    informe = mejor_metodo = None
    metodo_actual = "vandermonde"

    if request.method == "POST":
        metodo = request.form.get("metodo")
        metodo_actual = metodo
        try:
            vector_x = eval(request.form.get("vector_x"))
            vector_y = eval(request.form.get("vector_y"))

            if metodo == "vandermonde":
                resultado_vandermonde = vandermonde(vector_x, vector_y)
                resultado_vandermonde['solucion'] = pol_to_str(resultado_vandermonde['solucion'])
                if request.form.get("generar_informe"):
                    x_real = float(request.form.get("x_real", 0))
                    y_real = float(request.form.get("y_real", 0))
                    informe, mejor_metodo = generar_informe_comparativo(vector_x, vector_y, x_real, y_real)
            
            elif metodo == "newtoninter":
                resultado_newton = newtoninter(vector_x, vector_y)
                resultado_newton['solucion'] = pol_to_str(resultado_newton['solucion'])
                if request.form.get("generar_informe"):
                    x_real = float(request.form.get("x_real", 0))
                    y_real = float(request.form.get("y_real", 0))
                    informe, mejor_metodo = generar_informe_comparativo(vector_x, vector_y, x_real, y_real)
            
            elif metodo == "lagrange":
                resultado_lagrange = lagrange(vector_x, vector_y)
                resultado_lagrange['solucion'] = pol_to_str(resultado_lagrange['solucion'])
                if request.form.get("generar_informe"):
                    x_real = float(request.form.get("x_real", 0))
                    y_real = float(request.form.get("y_real", 0))
                    informe, mejor_metodo = generar_informe_comparativo(vector_x, vector_y, x_real, y_real)
            
            elif metodo == "splinel":
                resultado_spline = spline_interpolation(vector_x,vector_y,1)
                if request.form.get("generar_informe"):
                    x_real = float(request.form.get("x_real", 0))
                    y_real = float(request.form.get("y_real", 0))
                    informe, mejor_metodo = generar_informe_comparativo(vector_x, vector_y, x_real, y_real)

            elif metodo == "splinec":
                resultado_spline = spline_interpolation(vector_x,vector_y,3)
                if request.form.get("generar_informe"):
                    x_real = float(request.form.get("x_real", 0))
                    y_real = float(request.form.get("y_real", 0))
                    informe, mejor_metodo = generar_informe_comparativo(vector_x, vector_y, x_real, y_real)

        except Exception as e:
            if metodo == "vandermonde":
                error_vandermonde = str(e)
            elif metodo == "newtoninter":
                error_newton = str(e)
            elif metodo == "lagrange":
                error_lagrange = str(e)

    return render_template("chapter3.html",
                            metodo_actual=metodo_actual,
                            resultado_vandermonde=resultado_vandermonde,
                            error_vandermonde=error_vandermonde,
                            resultado_newton=resultado_newton,
                            error_newton=error_newton,
                            resultado_lagrange=resultado_lagrange,
                            error_lagrange=error_lagrange,
                            resultado_spline=resultado_spline,
                            error_spline=error_spline,
                            informe=informe,
                            mejor_metodo=mejor_metodo)

if __name__ == "__main__":
    app.run(debug=True)