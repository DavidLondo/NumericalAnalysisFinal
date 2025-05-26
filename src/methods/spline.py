import numpy as np
import matplotlib.pyplot as plt
import base64
import io

def spline_interpolation(x_vals, y_vals, grado):
    try:
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)
        n = len(x)

        if len(x) != len(y):
            return {"error": "Los vectores x e y deben tener la misma longitud."}
        if n < 2:
            return {"error": "Se requieren al menos dos puntos para interpolar."}
        if grado not in [1, 2, 3]:
            return {"error": "El grado debe ser 1 (lineal), 2 (cuadrático) o 3 (cúbico)."}

        d = grado
        num_eqs = (d + 1) * (n - 1)
        A = np.zeros((num_eqs, num_eqs))
        b = np.zeros(num_eqs)

        c = 0
        h = 0

        # Sistemas para condiciones
        if d == 1:
            for i in range(n - 1):
                A[h, c] = x[i]
                A[h, c + 1] = 1
                b[h] = y[i]
                h += 1
                c += 2

            c = 0
            for i in range(1, n):
                A[h, c] = x[i]
                A[h, c + 1] = 1
                b[h] = y[i]
                h += 1
                c += 2

        elif d == 2:
            c = 0
            for i in range(n - 1):
                A[h, c] = x[i] ** 2
                A[h, c + 1] = x[i]
                A[h, c + 2] = 1
                b[h] = y[i]
                h += 1
                c += 3

            c = 0
            for i in range(1, n):
                A[h, c] = x[i] ** 2
                A[h, c + 1] = x[i]
                A[h, c + 2] = 1
                b[h] = y[i]
                h += 1
                c += 3

            c = 0
            for i in range(1, n - 1):
                A[h, c] = 2 * x[i]
                A[h, c + 1] = 1
                A[h, c + 3] = -2 * x[i]
                A[h, c + 4] = -1
                b[h] = 0
                h += 1
                c += 3

            A[h, 0] = 2  # segunda derivada nula al inicio
            b[h] = 0

        elif d == 3:
            c = 0
            for i in range(n - 1):
                A[h, c] = x[i] ** 3
                A[h, c + 1] = x[i] ** 2
                A[h, c + 2] = x[i]
                A[h, c + 3] = 1
                b[h] = y[i]
                h += 1
                c += 4

            c = 0
            for i in range(1, n):
                A[h, c] = x[i] ** 3
                A[h, c + 1] = x[i] ** 2
                A[h, c + 2] = x[i]
                A[h, c + 3] = 1
                b[h] = y[i]
                h += 1
                c += 4

            c = 0
            for i in range(1, n - 1):
                A[h, c] = 3 * x[i] ** 2
                A[h, c + 1] = 2 * x[i]
                A[h, c + 2] = 1
                A[h, c + 4] = -3 * x[i] ** 2
                A[h, c + 5] = -2 * x[i]
                A[h, c + 6] = -1
                b[h] = 0
                h += 1
                c += 4

            c = 0
            for i in range(1, n - 1):
                A[h, c] = 6 * x[i]
                A[h, c + 1] = 2
                A[h, c + 4] = -6 * x[i]
                A[h, c + 5] = -2
                b[h] = 0
                h += 1
                c += 4

            # condiciones de frontera
            A[h, 0] = 6 * x[0]
            A[h, 1] = 2
            b[h] = 0
            h += 1

            A[h, -4] = 6 * x[-1]
            A[h, -3] = 2
            b[h] = 0

        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return {"error": "El sistema de ecuaciones no tiene solución única."}

        tabla = coef.reshape(n - 1, d + 1)

        # ----------- GRAFICAR ------------
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, 'ro', label="Datos")

        for i in range(n - 1):
            x_plot = np.linspace(x[i], x[i + 1], 100)
            if d == 1:
                y_plot = tabla[i, 0] * x_plot + tabla[i, 1]
            elif d == 2:
                y_plot = tabla[i, 0] * x_plot ** 2 + tabla[i, 1] * x_plot + tabla[i, 2]
            elif d == 3:
                y_plot = (tabla[i, 0] * x_plot ** 3 +
                          tabla[i, 1] * x_plot ** 2 +
                          tabla[i, 2] * x_plot +
                          tabla[i, 3])
            ax.plot(x_plot, y_plot, label=f"Tramo {i + 1}")

        ax.set_title("Interpolación por Spline")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        grafica_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {
            "solucion": tabla.tolist(),
            "grafica_base64": grafica_base64
        }

    except Exception as e:
        return {"error": f"Ocurrió un error inesperado: {str(e)}"}
