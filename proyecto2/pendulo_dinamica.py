import numpy as np
import matplotlib.pyplot as plt

# --- Definición de las Ecuaciones Diferenciales ---
# x1 = theta
# x2 = d(theta)/dt
# dx1/dt = x2
# dx2/dt = -nu*x2 - sin(x1) + tau*sin(x3)
# dx3/dt = 2*pi*f

def fun1(x1, x2, x3, t, nu, tau, f):
    """ Ecuación para dx1/dt """
    return x2

def fun2(x1, x2, x3, t, nu, tau, f):
    """ Ecuación para dx2/dt """
    return -nu * x2 - np.sin(x1) + tau * np.sin(x3)

def fun3(x1, x2, x3, t, nu, tau, f):
    """ Ecuación para dx3/dt """
    return 2 * np.pi * f

# --- Lectura de Datos de Entrada ---
with open("input_data.inp", "r") as file:
    lines = file.readlines()

# Lee parámetros de simulación y condiciones iniciales
n, t0, x10, x20, x30, h = map(float, lines[1].strip().split())
n = int(n)

# Lee parámetros del sistema físico
nu, tau, f = map(float, lines[3].strip().split())

# --- Inicialización de Arreglos ---
x = np.zeros((n, 3))
t = np.zeros(n)
k = np.zeros((4, 3))

# Establece condiciones iniciales
t[0] = t0
x[0, 0] = x10
x[0, 1] = x20
x[0, 2] = x30 # Corresponde a 2*pi*f*t0

# --- Solución Numérica (Runge-Kutta 4to orden) ---
# Este bucle es una adaptación directa del Código Python 2.6
for i in range(1, n):
    # Variables en el paso anterior
    x1_i, x2_i, x3_i = x[i-1, 0], x[i-1, 1], x[i-1, 2]
    t_i = t[i-1]

    # Cálculo de las pendientes intermedias (k1, k2, k3, k4)
    k[0, 0] = fun1(x1_i, x2_i, x3_i, t_i, nu, tau, f)
    k[0, 1] = fun2(x1_i, x2_i, x3_i, t_i, nu, tau, f)
    k[0, 2] = fun3(x1_i, x2_i, x3_i, t_i, nu, tau, f)

    k[1, 0] = fun1(x1_i + h*k[0,0]/2.0, x2_i + h*k[0,1]/2.0, x3_i + h*k[0,2]/2.0, t_i + h/2.0, nu, tau, f)
    k[1, 1] = fun2(x1_i + h*k[0,0]/2.0, x2_i + h*k[0,1]/2.0, x3_i + h*k[0,2]/2.0, t_i + h/2.0, nu, tau, f)
    k[1, 2] = fun3(x1_i + h*k[0,0]/2.0, x2_i + h*k[0,1]/2.0, x3_i + h*k[0,2]/2.0, t_i + h/2.0, nu, tau, f)

    k[2, 0] = fun1(x1_i + h*k[1,0]/2.0, x2_i + h*k[1,1]/2.0, x3_i + h*k[1,2]/2.0, t_i + h/2.0, nu, tau, f)
    k[2, 1] = fun2(x1_i + h*k[1,0]/2.0, x2_i + h*k[1,1]/2.0, x3_i + h*k[1,2]/2.0, t_i + h/2.0, nu, tau, f)
    k[2, 2] = fun3(x1_i + h*k[1,0]/2.0, x2_i + h*k[1,1]/2.0, x3_i + h*k[1,2]/2.0, t_i + h/2.0, nu, tau, f)
    
    k[3, 0] = fun1(x1_i + h*k[2,0], x2_i + h*k[2,1], x3_i + h*k[2,2], t_i + h, nu, tau, f)
    k[3, 1] = fun2(x1_i + h*k[2,0], x2_i + h*k[2,1], x3_i + h*k[2,2], t_i + h, nu, tau, f)
    k[3, 2] = fun3(x1_i + h*k[2,0], x2_i + h*k[2,1], x3_i + h*k[2,2], t_i + h, nu, tau, f)
    
    # Actualiza las variables x y el tiempo t
    for j in range(3):
        x[i, j] = x[i-1, j] + h * (k[0, j] + 2.0*k[1, j] + 2.0*k[2, j] + k[3, j]) / 6.0
    
    t[i] = t[i-1] + h

# --- Creación de Gráficas ---
# Esta sección usa la misma estructura de subplots del Código Python 2.6
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Primera gráfica: Ángulo vs Tiempo
axes[0].plot(t, x[:, 0], color="blue", linewidth=0.8)
axes[0].set_xlabel("Tiempo, $t$")
axes[0].set_ylabel("Ángulo, $x_1(t) = \\theta(t)$")
axes[0].set_title("Dinámica del Péndulo vs Tiempo")
axes[0].grid(True)

# Segunda gráfica: Espacio Fase (Velocidad angular vs Ángulo)
axes[1].plot(x[:, 0], x[:, 1], color="red", linewidth=0.5)
axes[1].set_xlabel("Ángulo, $x_1(t) = \\theta(t)$")
axes[1].set_ylabel("Velocidad Angular, $x_2(t)$")
axes[1].set_title("Órbita en Espacio Fase")
axes[1].grid(True)

# Ajusta el diseño para prevenir traslape de títulos
plt.tight_layout()

# Muestra ambas gráficas
plt.show()
