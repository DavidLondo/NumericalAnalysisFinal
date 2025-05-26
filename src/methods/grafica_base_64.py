import matplotlib.pyplot as plt
import matplotlib
import io
import base64
matplotlib.use('agg')

def generar_grafica_base64(x, y, xpol, ypol, titulo, xreal=None, yreal=None):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'ro', label='Datos originales')
    ax.plot(xpol, ypol, 'b-', label='Polinomio interpolante')
    
    if xreal is not None and yreal is not None:
        ax.plot(xreal, yreal, 'g--', label='Función real')
    
    ax.grid(True)
    ax.set_title(titulo)
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    imagen_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return imagen_base64