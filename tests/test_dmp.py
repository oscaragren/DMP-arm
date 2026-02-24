import numpy as np
from dmp.dmp import fit, rollout_simple, rollout_rk4, DMPModel

def test_single_simple_demo_reproduction():

    dt = 0.01
    tau = 1.0
    T = int(tau / dt) + 1
    t = np.linspace(0, tau, T)

    # Simple demo: one joint sine, others linear
    q_demo = np.zeros((T, 5)) # 5 DoF
    q_demo[:, 0] = 0.5 * np.sin(np.pi * t / tau) + 0.5
    
    for j in range(1, 5):
        q_demo[:, j] = np.linspace(0, 1, T)

    model = fit([q_demo], tau=tau, dt=dt, n_basis_functions=15, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen = rollout_simple(model, q_demo[0], q_demo[-1], tau=tau, dt=dt)
    
    rmse = np.sqrt(np.mean((q_gen - q_demo)**2, axis=0))
    
    assert np.all(rmse < 0.1), f"RMSE too high: {rmse}"
    #print(f"RMSE: {rmse}")

def test_single_rk4_demo_reproduction():

    dt = 0.01
    tau = 1.0
    T = int(tau / dt) + 1
    t = np.linspace(0, tau, T)

    # Simple demo: one joint sine, others linear
    q_demo = np.zeros((T, 5)) # 5 DoF
    q_demo[:, 0] = 0.5 * np.sin(np.pi * t / tau) + 0.5
    
    for j in range(1, 5):
        q_demo[:, j] = np.linspace(0, 1, T)

    model = fit([q_demo], tau=tau, dt=dt, n_basis_functions=15, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen = rollout_rk4(model, q_demo[0], q_demo[-1], tau=tau, dt=dt)
    
    rmse = np.sqrt(np.mean((q_gen - q_demo)**2, axis=0))

    assert np.all(rmse < 0.1), f"RMSE too high: {rmse}"
    #print(f"RMSE: {rmse}")

if __name__ == "__main__":
    test_single_simple_demo_reproduction()
    #test_single_rk4_demo_reproduction()