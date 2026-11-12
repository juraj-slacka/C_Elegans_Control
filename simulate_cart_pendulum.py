"""
Simulate and Benchmark Inverted Cart-Pendulum Controllers:
1. Biological C. elegans Connectome (279 neurons, 97.97% sparse)
2. Scale-Matched Multi-Layer Perceptron (MLP)
3. Long Short-Term Memory (LSTM)
4. Saturated Linear Quadratic Regulator (LQR)

Uses the pre-trained model weights stored in models/
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from src.cart_pole_env import CartPendulumEnv
from src.models import (
    CElegansFullConnectomeController,
    MLPController,
    LSTMController,
    CartPendulumLQRController
)


def run_simulation():
    device = 'cpu'
    dt = 0.02
    steps = 500
    time_arr = np.linspace(0.0, steps * dt, steps)
    os.makedirs('outputs', exist_ok=True)

    print("=================================================================")
    print("       INVERTED CART-PENDULUM SIMULATION (SAVED MODELS)")
    print("=================================================================")

    # 1. Load Pre-trained Controllers
    models_dir = "models"
    
    fc = CElegansFullConnectomeController()
    fc.load_state_dict(torch.load(os.path.join(models_dir, "c_elegans_full_connectome.pt"), map_location=device))
    fc.eval()
    
    mlp = MLPController()
    mlp.load_state_dict(torch.load(os.path.join(models_dir, "mlp_controller.pt"), map_location=device))
    mlp.eval()
    
    lstm = LSTMController()
    lstm.load_state_dict(torch.load(os.path.join(models_dir, "lstm_controller.pt"), map_location=device))
    lstm.eval()
    
    lqr = CartPendulumLQRController(dt=dt)

    controllers = {
        'Connectome (279)': ('#1f77b4', '-', fc),
        'MLP (279)': ('#ff7f0e', '--', mlp),
        'LSTM (H=64)': ('#9467bd', '-.', lstm),
        'Saturated LQR': ('#2ca02c', ':', lqr)
    }

    # -----------------------------------------------------------------
    # Test 1: Severe Initial Tilt Recovery (theta0 = 0.20 rad / 11.46 deg)
    # -----------------------------------------------------------------
    print("\n--- Scenario 1: Severe Initial Tilt Recovery (theta0 = 11.46 deg) ---")
    results_tilt = {}
    
    for name, (col, ls, ctrl) in controllers.items():
        env = CartPendulumEnv(dt=dt, max_steps=steps)
        obs = env.reset(initial_state=[-0.30, 0.0, 0.20, 0.0])
        hidden = ctrl.init_hidden(1, device=device) if hasattr(ctrl, 'init_hidden') else None
        
        thetas, xs, us = [], [], []
        for _ in range(steps):
            if name == 'Saturated LQR':
                u = ctrl.compute_action(obs)
            else:
                with torch.no_grad():
                    act, hidden = ctrl.forward_step(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), hidden)
                u = float(act.item())
            obs, _, _, _, info = env.step(u)
            thetas.append(np.rad2deg(obs[2]))
            xs.append(obs[0])
            us.append(info['u'])
            
        us = np.array(us)
        thetas = np.array(thetas)
        xs = np.array(xs)
        
        peak_f = float(np.abs(us).max())
        effort = float(np.sum(us ** 2) * dt)
        
        # Settling time (|theta| < 0.5 deg and |x| < 0.05 m)
        settled_idx = np.where((np.abs(thetas) < 0.5) & (np.abs(xs) < 0.05))[0]
        ts = steps * dt
        for i in range(len(thetas) - 25):
            if np.all(np.abs(thetas[i:]) < 0.5) and np.all(np.abs(xs[i:]) < 0.05):
                ts = i * dt
                break
                
        results_tilt[name] = {
            'peak_force': peak_f,
            'effort': effort,
            'settling_time': ts,
            'thetas': thetas,
            'xs': xs,
            'us': us
        }
        print(f"[{name:17s}] Peak Force: {peak_f:5.2f} N | Effort (∫u²dt): {effort:6.2f} N²s | Settling: {ts:4.2f} s")

    # -----------------------------------------------------------------
    # Test 2: External Impulse Shock (+5.0 N for 0.1 s at t = 2.0 s)
    # -----------------------------------------------------------------
    print("\n--- Scenario 2: External Impulse Kick (+5.0 N at t = 2.0 s) ---")
    results_shock = {}
    
    for name, (col, ls, ctrl) in controllers.items():
        env = CartPendulumEnv(dt=dt, max_steps=steps)
        obs = env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        hidden = ctrl.init_hidden(1, device=device) if hasattr(ctrl, 'init_hidden') else None
        
        thetas, us = [], []
        for step in range(steps):
            t = step * dt
            # Add impulse disturbance
            dist = 5.0 if (2.0 <= t < 2.1) else 0.0
            
            if name == 'Saturated LQR':
                u = ctrl.compute_action(obs)
            else:
                with torch.no_grad():
                    act, hidden = ctrl.forward_step(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), hidden)
                u = float(act.item())
            obs, _, _, _, info = env.step(u + dist)
            thetas.append(np.rad2deg(obs[2]))
            us.append(info['u'])
            
        thetas = np.array(thetas)
        max_dev = float(np.abs(thetas[int(2.0/dt):]).max())
        results_shock[name] = {'max_dev': max_dev, 'thetas': thetas, 'us': np.array(us)}
        print(f"[{name:17s}] Max Impulse Angular Deviation: {max_dev:5.2f} deg")

    # -----------------------------------------------------------------
    # Plot & Save Simulation Figure
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    
    # 1. Tilt angle
    for name, (col, ls, _) in controllers.items():
        axes[0].plot(time_arr, results_tilt[name]['thetas'], label=name, color=col, linestyle=ls, lw=1.8)
    axes[0].set_ylabel('Tilt θ [deg]', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Inverted Cart-Pendulum: Severe Initial Tilt Recovery (θ₀ = 11.46°)', fontweight='bold')

    # 2. Cart position
    for name, (col, ls, _) in controllers.items():
        axes[1].plot(time_arr, results_tilt[name]['xs'], label=name, color=col, linestyle=ls, lw=1.8)
    axes[1].set_ylabel('Cart Position x [m]', fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. Control Force
    for name, (col, ls, _) in controllers.items():
        axes[2].plot(time_arr, results_tilt[name]['us'], label=name, color=col, linestyle=ls, lw=1.8)
    axes[2].axhline(20.0, color='red', linestyle='--', alpha=0.5, label='Saturation Ceiling (±20 N)')
    axes[2].axhline(-20.0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Force u(t) [N]', fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].set_ylim(-22, 22)

    plt.tight_layout()
    out_fig = 'outputs/cart_pendulum_simulation.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\n[Done] Simulation plot saved to: {out_fig}\n")


if __name__ == '__main__':
    run_simulation()
