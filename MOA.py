import numpy as np
import matplotlib.pyplot as plt

def solve_damped_oscillator(m, k, b, x0, v0, t_final, h):
    """
    Resuelve la ecuación del oscilador armónico amortiguado usando un método numérico.
    Implementa el algoritmo de Euler hacia adelante.

    Parámetros:
    m (float): Masa del oscilador (kg)
    k (float): Constante del resorte (N/m)
    b (float): Coeficiente de fricción (kg/s)
    x0 (float): Posición inicial (m)
    v0 (float): Velocidad inicial (m/s)
    t_final (float): Tiempo final de la simulación (s)
    h (float): Paso de tiempo (s)

    Retorna:
    tuple: (t, x, v) - arreglos de numpy para tiempo, posición y velocidad.
    """
    # 1. Inicialización de arreglos
    n_steps = int(t_final / h)
    t = np.linspace(0, t_final, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    a = np.zeros(n_steps + 1)

    # 2. Aplicar condiciones iniciales
    x[0] = x0
    v[0] = v0

    # 3. Bucle de iteración temporal
    for i in range(n_steps):
        # Ecuación (1.51): Calcular la aceleración en el paso actual i
        a[i] = -(b / m) * v[i] - (k / m) * x[i]
        
        # Ecuación (1.55): Calcular la velocidad en el siguiente paso i+1
        v[i+1] = v[i] + a[i] * h
        
        # Ecuación para actualizar la posición (método de Euler)
        x[i+1] = x[i] + v[i] * h

    return t, x, v

def calculate_energy(b, t_array, v_array, t_points):
    """
    Calcula la energía disipada en puntos específicos del tiempo.

    Parámetros:
    b (float): Coeficiente de fricción
    t_array (np.array): Arreglo de tiempo de la simulación
    v_array (np.array): Arreglo de velocidad de la simulación
    t_points (list): Lista de tiempos en los que calcular la energía
    """
    print(f"Energía disipada para b = {b:.2f} kg/s:")
    for t_final in t_points:
        # Encontrar el índice más cercano al tiempo final deseado
        idx = (np.abs(t_array - t_final)).argmin()
        
        # Seleccionar los datos hasta ese punto
        t_interval = t_array[:idx+1]
        v_interval = v_array[:idx+1]
        
        # El integrando para la Ecuación (1.57) es v^2(t)
        integrand = v_interval**2
        
        # Calcular la integral usando la regla del trapecio de numpy
        integral_val = np.trapz(integrand, t_interval)
        
        # Ecuación (1.57): Calcular la energía disipada
        E_dis = b * integral_val
        
        print(f"  - En t = {t_final:.1f} s, E_disipada = {E_dis:.4f} Joules")

# --- Parámetros del Problema ---
m = 0.5  # kg
k = 4.0  # N/m
# Valor de amortiguamiento crítico: b_c = 2*sqrt(m*k) = 2*sqrt(0.5*4) ~= 2.828
b_values_corrected = {
    "Subamortiguado (b=0.5)": 0.5,
    "Crítico (b=2.83)": 2 * np.sqrt(m * k),
    "Sobreamortiguado (b=5.0)": 5.0
}

# Condiciones iniciales
x0 = 1.0  # m
v0 = 0.0  # m/s

# Parámetros de la simulación
t_final = 10.0 # s
h = 0.01     # s

# --- Ejecución y Visualización ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

for label, b in b_values_corrected.items():
    # Resolver el sistema para cada valor de b
    t, x, v = solve_damped_oscillator(m, k, b, x0, v0, t_final, h)
    
    # Graficar la posición en función del tiempo
    ax.plot(t, x, label=f'{label}')
    
    # Calcular y mostrar la energía disipada
    print(f"--- Análisis para el caso: {label} ---")
    calculate_energy(b, t, v, t_points=[1.0, 2.0, 5.0])
    print("-" * 40)

# Configuración final del gráfico
ax.set_title(f'Movimiento del Oscilador Armónico Amortiguado (m={m} kg, k={k} N/m)', fontsize=16)
ax.set_xlabel('Tiempo (s)', fontsize=12)
ax.set_ylabel('Posición x(t) (m)', fontsize=12)
ax.legend(fontsize=10)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--') # Línea de equilibrio

plt.tight_layout()
plt.savefig("oscilador_amortiguado.png")

print("\nGráfico 'oscilador_amortiguado.png' generado exitosamente.")
