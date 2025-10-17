import numpy as np
import matplotlib.pyplot as plt

# --- Definición de las Ecuaciones Diferenciales (igual que antes) ---
def fun1(x1, x2, x3, t, nu, tau, f):
    return x2

def fun2(x1, x2, x3, t, nu, tau, f):
    return -nu * x2 - np.sin(x1) + tau * np.sin(x3)

def fun3(x1, x2, x3, t, nu, tau, f):
    return 2 * np.pi * f

# --- Parámetros de Simulación y del Sistema ---
n = 25000
t0 = 0.0
h = 0.01
nu = 0.5
tau = 1.15
f = 1.0 / (6.0 * np.pi) 

# --- Condiciones Iniciales ---
# Simulación 1
x10_a, x20_a, x30_a = 0.0, 0.0, 0.0
# Simulación 2 (con una pequeña perturbación en el ángulo inicial)
x10_b, x20_b, x30_b = 0.0001, 0.0, 0.0

# --- Inicialización de Arreglos ---
# Arreglos para la primera simulación ('a')
x_a = np.zeros((n, 3))
t = np.zeros(n) # El tiempo es el mismo para ambas
x_a[0,:] = [x10_a, x20_a, x30_a]

# Arreglos para la segunda simulación ('b')
x_b = np.zeros((n, 3))
x_b[0,:] = [x10_b, x20_b, x30_b]

t[0] = t0
k_a = np.zeros((4, 3))
k_b = np.zeros((4, 3))

# --- Solución Numérica para AMBAS trayectorias ---
for i in range(1, n):
    # --- Simulación A ---
    x1_a, x2_a, x3_a = x_a[i-1, 0], x_a[i-1, 1], x_a[i-1, 2]
    t_i = t[i-1]
    
    k_a[0, 0] = fun1(x1_a, x2_a, x3_a, t_i, nu, tau, f)
    k_a[0, 1] = fun2(x1_a, x2_a, x3_a, t_i, nu, tau, f)
    k_a[0, 2] = fun3(x1_a, x2_a, x3_a, t_i, nu, tau, f)
    # ... (Cálculos de k2, k3, k4 para 'a') ...
    k_a[1, 0] = fun1(x1_a + h*k_a[0,0]/2., x2_a + h*k_a[0,1]/2., x3_a + h*k_a[0,2]/2., t_i + h/2., nu, tau, f)
    k_a[1, 1] = fun2(x1_a + h*k_a[0,0]/2., x2_a + h*k_a[0,1]/2., x3_a + h*k_a[0,2]/2., t_i + h/2., nu, tau, f)
    k_a[1, 2] = fun3(x1_a + h*k_a[0,0]/2., x2_a + h*k_a[0,1]/2., x3_a + h*k_a[0,2]/2., t_i + h/2., nu, tau, f)
    k_a[2, 0] = fun1(x1_a + h*k_a[1,0]/2., x2_a + h*k_a[1,1]/2., x3_a + h*k_a[1,2]/2., t_i + h/2., nu, tau, f)
    k_a[2, 1] = fun2(x1_a + h*k_a[1,0]/2., x2_a + h*k_a[1,1]/2., x3_a + h*k_a[1,2]/2., t_i + h/2., nu, tau, f)
    k_a[2, 2] = fun3(x1_a + h*k_a[1,0]/2., x2_a + h*k_a[1,1]/2., x3_a + h*k_a[1,2]/2., t_i + h/2., nu, tau, f)
    k_a[3, 0] = fun1(x1_a + h*k_a[2,0], x2_a + h*k_a[2,1], x3_a + h*k_a[2,2], t_i + h, nu, tau, f)
    k_a[3, 1] = fun2(x1_a + h*k_a[2,0], x2_a + h*k_a[2,1], x3_a + h*k_a[2,2], t_i + h, nu, tau, f)
    k_a[3, 2] = fun3(x1_a + h*k_a[2,0], x2_a + h*k_a[2,1], x3_a + h*k_a[2,2], t_i + h, nu, tau, f)
    
    # --- Simulación B ---
    x1_b, x2_b, x3_b = x_b[i-1, 0], x_b[i-1, 1], x_b[i-1, 2]

    k_b[0, 0] = fun1(x1_b, x2_b, x3_b, t_i, nu, tau, f)
    k_b[0, 1] = fun2(x1_b, x2_b, x3_b, t_i, nu, tau, f)
    k_b[0, 2] = fun3(x1_b, x2_b, x3_b, t_i, nu, tau, f)
    # ... (Cálculos de k2, k3, k4 para 'b') ...
    k_b[1, 0] = fun1(x1_b + h*k_b[0,0]/2., x2_b + h*k_b[0,1]/2., x3_b + h*k_b[0,2]/2., t_i + h/2., nu, tau, f)
    k_b[1, 1] = fun2(x1_b + h*k_b[0,0]/2., x2_b + h*k_b[0,1]/2., x3_b + h*k_b[0,2]/2., t_i + h/2., nu, tau, f)
    k_b[1, 2] = fun3(x1_b + h*k_b[0,0]/2., x2_b + h*k_b[0,1]/2., x3_b + h*k_b[0,2]/2., t_i + h/2., nu, tau, f)
    k_b[2, 0] = fun1(x1_b + h*k_b[1,0]/2., x2_b + h*k_b[1,1]/2., x3_b + h*k_b[1,2]/2., t_i + h/2., nu, tau, f)
    k_b[2, 1] = fun2(x1_b + h*k_b[1,0]/2., x2_b + h*k_b[1,1]/2., x3_b + h*k_b[1,2]/2., t_i + h/2., nu, tau, f)
    k_b[2, 2] = fun3(x1_b + h*k_b[1,0]/2., x2_b + h*k_b[1,1]/2., x3_b + h*k_b[1,2]/2., t_i + h/2., nu, tau, f)
    k_b[3, 0] = fun1(x1_b + h*k_b[2,0], x2_b + h*k_b[2,1], x3_b + h*k_b[2,2], t_i + h, nu, tau, f)
    k_b[3, 1] = fun2(x1_b + h*k_b[2,0], x2_b + h*k_b[2,1], x3_b + h*k_b[2,2], t_i + h, nu, tau, f)
    k_b[3, 2] = fun3(x1_b + h*k_b[2,0], x2_b + h*k_b[2,1], x3_b + h*k_b[2,2], t_i + h, nu, tau, f)
    
    # --- Actualización de Variables ---
    t[i] = t[i-1] + h
    for j in range(3):
        x_a[i, j] = x_a[i-1, j] + h * (k_a[0,j] + 2.*k_a[1,j] + 2.*k_a[2,j] + k_a[3,j]) / 6.0
        x_b[i, j] = x_b[i-1, j] + h * (k_b[0,j] + 2.*k_b[1,j] + 2.*k_b[2,j] + k_b[3,j]) / 6.0

# --- Cálculo de la Diferencia ---
# Calcula la diferencia absoluta entre las trayectorias angulares (x1)
delta_x1 = np.abs(x_a[:, 0] - x_b[:, 0])

# --- Creación de Gráfica ---
plt.figure(figsize=(10, 6))
plt.plot(t, delta_x1, color='purple', label='Diferencia $|x_1(t) - x\'_1(t)|$')
plt.xlabel("Tiempo, $t$")
plt.ylabel("Diferencia Angular, $\\Delta x_1$")
plt.title("Sensibilidad a las Condiciones Iniciales")
# Usar una escala logarítmica es ideal para ver el crecimiento exponencial del caos
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
