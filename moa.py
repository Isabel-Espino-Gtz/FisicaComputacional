import numpy as np
import matplotlib.pyplot as plt

def simulate_oscillator(m, k, b, x0=1.0, v0=0.0, h=1e-3, tmax=5.0):
    """Simula el oscilador amortiguado con paso h hasta tmax usando esquema explícito."""
    N = int(np.round(tmax / h))
    t = np.linspace(0.0, N*h, N+1)
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    x[0] = x0
    v[0] = v0
    for i in range(N):
        a = -(b/m)*v[i] - (k/m)*x[i]           # a_i = -(b/m) v_i - (k/m) x_i
        x[i+1] = x[i] + v[i]*h + 0.5*a*h*h    # Ecuación (1.54)
        v[i+1] = v[i] + a*h                   # Ecuación (1.55)
    return t, x, v

def energy_dissipated_by_time(t, v, b, tf):
    """
    Calcula energía disipada hasta tiempo tf:
      E_dis(tf) = b * integral_0^{tf} v(t')^2 dt'
    tf fuera del rango se recorta al máximo disponible.
    """
    if tf <= t[0]:
        return 0.0
    # índice más grande tal que t[idx] <= tf
    idx = np.searchsorted(t, tf, side="right") - 1
    idx = max(0, min(idx, len(t)-1))
    # integral de v^2 desde t[0] hasta t[idx]
    integral_v2 = np.trapz(v[:idx+1]**2, t[:idx+1])
    return b * integral_v2

def mechanical_energy(m, k, x, v):
    """Energía mecánica total en cada instante: 1/2 m v^2 + 1/2 k x^2"""
    return 0.5*m*v**2 + 0.5*k*x**2

def analytic_solution(m, k, b, x0, v0, t):

    gamma = b / (2.0 * m)
    omega0 = np.sqrt(k / m)
    t = np.asarray(t)
    if gamma < omega0 - 0.0:
        # Subamortiguado
        omega_d = np.sqrt(omega0**2 - gamma**2)
        C1 = x0
        C2 = (v0 + gamma * x0) / omega_d
        x_a = np.exp(-gamma * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))
        # v_a: derivada de x_a
        v_a = np.exp(-gamma * t) * (
            -gamma * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))
            + (-C1 * omega_d * np.sin(omega_d * t) + C2 * omega_d * np.cos(omega_d * t))
        )
    elif np.isclose(gamma, omega0, atol=1e-12):
        # Críticamente amortiguado: x(t) = (C1 + C2 t) e^{-gamma t}
        C1 = x0
        C2 = v0 + gamma * x0
        x_a = (C1 + C2 * t) * np.exp(-gamma * t)
        v_a = (C2 - gamma * (C1 + C2 * t)) * np.exp(-gamma * t)
    else:
        # Sobreamortiguado: raíces reales r1,r2
        s = np.sqrt(gamma**2 - omega0**2)
        r1 = -gamma + s
        r2 = -gamma - s
        # x(t) = A e^{r1 t} + B e^{r2 t}
        # condiciones iniciales: x(0) = A + B = x0
        # v(0) = A r1 + B r2 = v0
        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        x_a = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        v_a = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)
    return x_a, v_a

def classification(b, m, k, tol=1e-6):
    """Clasifica régimen según b comparado con b_crit."""
    b_crit = 2.0 * np.sqrt(m * k)
    if np.isclose(b, b_crit, atol=tol):
        return "Críticamente amortiguado"
    elif b < b_crit:
        return "Subamortiguado"
    else:
        return "Sobreamortiguado"

def convergence_test(m, k, b, x0, v0, h_list, tmax=5.0, tf_check=5.0):
    """Prueba simple de convergencia: compara E_dis(tf_check) para distintas h."""
    print("\nPrueba de convergencia (E_dis en tf = {:.3f} s):".format(tf_check))
    header = f"{'h':>10} | {'E_dis(tf)':>12} | {'Residuo (J)':>12}"
    print(header)
    print("-" * len(header))
    prev = None
    for h in h_list:
        t, x, v = simulate_oscillator(m, k, b, x0=x0, v0=v0, h=h, tmax=tmax)
        E = mechanical_energy(m, k, x, v)
        E_dis = energy_dissipated_by_time(t, v, b, tf_check)
        resid = E[0] - (E[-1] + energy_dissipated_by_time(t, v, b, t[-1]))
        print(f"{h:10.3e} | {E_dis:12.6f} | {resid:12.6e}")
        prev = E_dis

if __name__ == "__main__":
    # Parámetros del proyecto
    m = 0.5
    k = 4.0
    b_crit = 2.0 * np.sqrt(k * m)   # coeficiente crítico
    b_list = [0.5, b_crit, 3.0]     # subam., crítico, sobream.
    x0 = 1.0
    v0 = 0.0
    h = 1e-3
    tmax = 5.0
    tf_list = [1.0, 2.0, 5.0]

    print("Simulación del oscilador amortiguado (m=0.5 kg, k=4 N/m).")
    print(f"Paso h = {h} s, simulación hasta tmax = {tmax} s.\n")
    header = f"{'b (kg/s)':>8} | {'Régimen':>22} | {'E_dis(1s) (J)':>12} | {'E_dis(2s) (J)':>12} | {'E_dis(5s) (J)':>12} | {'E_mech(0) (J)':>12} | {'Residuo (J)':>12}"
    print(header)
    print("-" * len(header))

    # guardamos simulaciones para reutilizarlas en las gráficas
    sims = {}
    for b in b_list:
        t, x, v = simulate_oscillator(m, k, b, x0=x0, v0=v0, h=h, tmax=tmax)
        E = mechanical_energy(m, k, x, v)

        # Energía disipada en los tf de interés
        E_dis_values = [energy_dissipated_by_time(t, v, b, tf) for tf in tf_list]
        E_mech_initial = E[0]

        # Comprobación energética total hasta tmax
        E_dis_total = energy_dissipated_by_time(t, v, b, t[-1])  # o tf=tmax
        residuo = E_mech_initial - (E[-1] + E_dis_total)

        # Clasificación del régimen
        regime = classification(b, m, k, tol=1e-9)

        sims[b] = {"t": t, "x": x, "v": v, "E": E, "E_dis_total": E_dis_total, "residuo": residuo}

        E1, E2, E5 = E_dis_values
        print(f"{b:8.6f} | {regime:>22} | {E1:12.6f} | {E2:12.6f} | {E5:12.6f} | {E_mech_initial:12.6f} | {residuo:12.6e}")


        try:
            x_a, v_a = analytic_solution(m, k, b, x0, v0, t)
            # error L2 y máximo
            err_L2 = np.sqrt(np.trapz((sims[b]["x"] - x_a)**2, t) / (t[-1] - t[0]))
            err_max = np.max(np.abs(sims[b]["x"] - x_a))
            print(f"           -> Error solución analítica (x): L2 = {err_L2:.3e}, max = {err_max:.3e}")
        except Exception as e:
            print("           -> No se pudo calcular solución analítica:", e)


    h_list = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
    convergence_test(m, k, b=0.5, x0=x0, v0=v0, h_list=h_list, tmax=tmax, tf_check=5.0)

    # === GRÁFICAS COMPARATIVAS (reutiliza sims) ===
    # Posición
    plt.figure(figsize=(8,4))
    for b, data in sims.items():
        plt.plot(data["t"], data["x"], label=f"b={b:.6f} kg/s")
    plt.title("Oscilador amortiguado: Posición x(t)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Posición x (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Velocidad
    plt.figure(figsize=(8,4))
    for b, data in sims.items():
        plt.plot(data["t"], data["v"], label=f"b={b:.6f} kg/s")
    plt.title("Oscilador amortiguado: Velocidad v(t)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad v (m/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Energía mecánica
    plt.figure(figsize=(8,4))
    for b, data in sims.items():
        plt.plot(data["t"], data["E"], label=f"b={b:.6f} kg/s")
    plt.title("Oscilador amortiguado: Energía mecánica E(t)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía mecánica (J)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    b_show = 0.5
    if b_show in sims:
        t = sims[b_show]["t"]
        x_num = sims[b_show]["x"]
        x_ana, v_ana = analytic_solution(m, k, b_show, x0, v0, t)
        plt.figure(figsize=(8,4))
        plt.plot(t, x_num, label="numérico")
        plt.plot(t, x_ana, "--", label="analítico")
        plt.title(f"Comparación numérico vs analítico (b={b_show})")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Posición x (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

