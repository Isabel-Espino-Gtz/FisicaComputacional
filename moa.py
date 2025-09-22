#!/usr/bin/env python3
"""
Simulación del oscilador amortiguado usando diferencias finitas (ecuaciones 1.54 y 1.55).
Genera tablas de resultados y gráficas de posición, velocidad y energía mecánica.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_oscillator(m, k, b, x0=1.0, v0=0.0, h=1e-3, tmax=5.0):
    """Simula el oscilador amortiguado con paso h hasta tmax."""
    N = int(np.round(tmax / h))
    t = np.linspace(0.0, N*h, N+1)
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    x[0] = x0
    v[0] = v0
    for i in range(N):
        a = -(b/m)*v[i] - (k/m)*x[i]    # a_i = -(b/m) v_i - (k/m) x_i
        x[i+1] = x[i] + v[i]*h + 0.5*a*h*h  # Ecuación (1.54)
        v[i+1] = v[i] + a*h               # Ecuación (1.55)
    return t, x, v

def energy_dissipated_by_time(t, v, b, tf):
    """Calcula energía disipada hasta tiempo tf."""
    idx = int(np.round(tf / (t[1]-t[0])))
    integral_v2 = np.trapz(v[:idx+1]**2, t[:idx+1])
    return b * integral_v2

def mechanical_energy(m, k, x, v):
    """Energía mecánica total en cada instante."""
    return 0.5*m*v**2 + 0.5*k*x**2

if __name__ == "__main__":
    # Parámetros del proyecto
    m = 0.5
    k = 4.0
    b_list = [0.5, 2.0, 3.0]
    x0 = 1.0
    v0 = 0.0
    h = 1e-3
    tmax = 5.0
    tf_list = [1.0, 2.0, 5.0]

    # Encabezado tabla
    print("Simulación del oscilador amortiguado (m=0.5 kg, k=4 N/m).")
    print(f"Paso h = {h} s, simulación hasta tmax = {tmax} s.\n")
    print(f"{'b (kg/s)':>8} | {'Régimen':>22} | {'E_dis(1s) (J)':>12} | {'E_dis(2s) (J)':>12} | {'E_dis(5s) (J)':>12} | {'E_mech(0) (J)':>12}")
    print("-"*95)

    for b in b_list:
        # Simulación
        t, x, v = simulate_oscillator(m, k, b, x0=x0, v0=v0, h=h, tmax=tmax)
        E = mechanical_energy(m, k, x, v)

        # Energía disipada
        E_dis_values = [energy_dissipated_by_time(t, v, b, tf) for tf in tf_list]
        E_mech_initial = E[0]

        # Clasificación del régimen
        criterio = 2.0*np.sqrt(k*m)
        if abs(b - criterio) < 1e-8:
            regime = "Críticamente amortiguado (≈)"
        elif b < criterio:
            regime = "Subamortiguado"
        else:
            regime = "Sobreamortiguado"

        E1, E2, E5 = E_dis_values
        print(f"{b:8.3f} | {regime:>22} | {E1:12.6f} | {E2:12.6f} | {E5:12.6f} | {E_mech_initial:12.6f}")

        # === GRÁFICAS ===
        # Posición
        plt.figure(figsize=(8,4))
        plt.plot(t, x)
        plt.title(f"Oscilador amortiguado: Posición x(t), b={b} kg/s")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Posición x (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Velocidad
        plt.figure(figsize=(8,4))
        plt.plot(t, v)
        plt.title(f"Oscilador amortiguado: Velocidad v(t), b={b} kg/s")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Velocidad v (m/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Energía mecánica
        plt.figure(figsize=(8,4))
        plt.plot(t, E)
        plt.title(f"Oscilador amortiguado: Energía mecánica E(t), b={b} kg/s")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Energía mecánica (J)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # === GRÁFICAS COMPARATIVAS ===
# Posición comparativa
        plt.figure(figsize=(8,4))
        for b in b_list:
            t, x, v = simulate_oscillator(m, k, b, x0=x0, v0=v0, h=h, tmax=tmax)
            plt.plot(t, x, label=f"b={b} kg/s")
        plt.title("Oscilador amortiguado: Posición x(t)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Posición x (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Velocidad comparativa
        plt.figure(figsize=(8,4))
        for b in b_list:
            t, x, v = simulate_oscillator(m, k, b, x0=x0, v0=v0, h=h, tmax=tmax)
            plt.plot(t, v, label=f"b={b} kg/s")
        plt.title("Oscilador amortiguado: Velocidad v(t)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Velocidad v (m/s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Energía mecánica comparativa
        plt.figure(figsize=(8,4))
        for b in b_list:
            t, x, v = simulate_oscillator(m, k, b, x0=x0, v0=v0, h=h, tmax=tmax)
            E = mechanical_energy(m, k, x, v)
            plt.plot(t, E, label=f"b={b} kg/s")
        plt.title("Oscilador amortiguado: Energía mecánica E(t)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Energía mecánica (J)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  
        plt.show()

