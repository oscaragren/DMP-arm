import numpy as np
from typing import Callable

Array = np.ndarray

def rk4_step(f: Callable[[float, Array], Array], t: float, y: Array, dt: float) -> Array:

    """
    One RK4 integration step for y' = f(t, y)
    """

    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)

    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def rk4_integrate_ode(f: Callable[[float, Array], Array], y0: Array, t0: float, t1: float, dt: float) -> Array:

    """
    Integrate ODE from t0 to t0 + num_steps*dt using RK4
    """

    n_steps = int(np.ceil((t1 - t0) / dt))
    ys = np.empty((n_steps + 1, y0.shape[0]), dtype=float)
    ts = np.empty(n_steps + 1, dtype=float)

    y = y0.astype(float).copy()
    t = float(t0)

    ys[0] = y
    ts[0] = t

    for i in range(1, n_steps + 1):
        dt_i = min(dt, t1 - t)
        y = rk4_step(f, t, y, dt_i)
        t += dt_i
        ys[i] = y
        ts[i] = t

    return ys, ts