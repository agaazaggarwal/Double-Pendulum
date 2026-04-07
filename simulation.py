import numpy as np
from scipy.integrate import solve_ivp
from config import L1, L2, M1, M2, G

def equations(t, y, L1, L2, M1, M2, G):
    th1, th2, w1, w2 = y
    delta = th2 - th1
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta)**2
    dw1 = (M2 * L1 * w1**2 * np.sin(delta) * np.cos(delta) +
           M2 * G * np.sin(th2) * np.cos(delta) +
           M2 * L2 * w2**2 * np.sin(delta) -
           (M1 + M2) * G * np.sin(th1)) / den1
    den2 = (L2 / L1) * den1
    dw2 = (-M2 * L2 * w2**2 * np.sin(delta) * np.cos(delta) +
           (M1 + M2) * G * np.sin(th1) * np.cos(delta) -
           (M1 + M2) * L1 * w1**2 * np.sin(delta) -
           (M1 + M2) * G * np.sin(th2)) / den2
    return [w1, w2, dw1, dw2]

def simulate_double_pendulum(th1_0, th2_0, w1_0, w2_0, t_eval):
    y0 = [th1_0, th2_0, w1_0, w2_0]
    sol = solve_ivp(equations, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                    args=(L1, L2, M1, M2, G), method='RK45')
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]