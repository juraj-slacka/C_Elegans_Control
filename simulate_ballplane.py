"""
Simulate and Benchmark Ball-on-a-Plane Controllers:
1. Biological C. elegans Connectome (279 neurons, 97.97% sparse)
2. Scale-Matched Multi-Layer Perceptron (MLP)
3. Long Short-Term Memory (LSTM)
4. Saturated Linear Quadratic Regulator with Integral Action (LQI)

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
from src.ballplane_env import BallPlaneEnv
from src.models import (
    BallPlaneConnectomeController,
    BallPlaneMLPController,
    BallPlaneLSTMController,
    BallPlaneLQRController
)


def simulate_plant(ctrl_name, ctrl, env, duration=6.0, target_func=None, dist_func=None, device='cpu'):
    dt = env.dt
    steps = int(duration / dt)
    time_arr = np.linspace(0.0, duration, steps)
    
    x_hist = np.zeros(steps)
    dx_hist = np.zeros(steps)
    theta_hist = np.zeros(steps)
    u_hist = np.zeros(steps)
    ref_hist = np.zeros(steps)
    
    if hasattr(ctrl, 'reset'):
        ctrl.reset()
    hidden = ctrl.init_hidden(1, device=device) if hasattr(ctrl, 'init_hidden') else None
    
    theta_max = env.theta_max
    x_scale = 0.15
    v_scale = 0.50
    
    for k in range(steps):
        t = time_arr[k]
        x_ref = target_func(t) if target_func is not None else 0.15
        ref_hist[k] = x_ref
        
        x, dx, theta = env.state
        x_hist[k] = x
        dx_hist[k] = dx
        theta_hist[k] = theta
        
        e_x = x - x_ref
        F_dist = dist_func(t) if dist_func is not None else 0.0
        
        if ctrl_name == 'Saturated LQR':
            u = ctrl.compute_action(x, dx, theta, x_ref)
        else:
            # Normalized observation vector: [e_x/x_scale, dx/v_scale, theta/theta_max, x_ref/x_scale]
            obs_np = np.array([e_x / x_scale, dx / v_scale, theta / theta_max, x_ref / x_scale], dtype=np.float32)
            obs_t = torch.tensor(obs_np, device=device).unsqueeze(0)
            with torch.no_grad():
                act, hidden = ctrl.forward_step(obs_t, hidden)
            u = float(act.item())
            
        u_hist[k] = u
        env.step(u, F_dist=F_dist)
        
    return {
        'time': time_arr,
        'x': x_hist,
        'dx': dx_hist,
        'theta': theta_hist,
        'u': u_hist,
        'ref': ref_hist
    }


def run_simulation():
    device = 'cpu'
    os.makedirs('outputs', exist_ok=True)
    models_dir = "models"

    print("=================================================================")
    print("         BALL-ON-A-PLANE SIMULATION (SAVED MODELS)")
    print("=================================================================")

    # Load Pre-trained Controllers
    fc = BallPlaneConnectomeController()
    fc.load_state_dict(torch.load(os.path.join(models_dir, "connectome_ballplane.pt"), map_location=device))
    fc.eval()
    
    mlp = BallPlaneMLPController()
    mlp.load_state_dict(torch.load(os.path.join(models_dir, "mlp_ballplane.pt"), map_location=device))
    mlp.eval()
    
    lstm = BallPlaneLSTMController()
    lstm.load_state_dict(torch.load(os.path.join(models_dir, "lstm_ballplane.pt"), map_location=device))
    lstm.eval()
    
    lqr = BallPlaneLQRController()

    controllers = {
        'Connectome (279)': ('#1f77b4', '-', fc),
        'MLP (279)': ('#ff7f0e', '--', mlp),
        'LSTM (H=64)': ('#9467bd', '-.', lstm),
        'Saturated LQR': ('#2ca02c', ':', lqr)
    }

    # -----------------------------------------------------------------
    # Test 1: Nominal 15 cm Step Tracking (t=1.0s, x_ref = 0 -> 0.15 m)
    # -----------------------------------------------------------------
    print("\n--- Scenario 1: Nominal 15 cm Step Response ---")
    step_res = {}
    
    def step_target(t):
        return 0.15 if t >= 1.0 else 0.0

    for name, (col, ls, ctrl) in controllers.items():
        env = BallPlaneEnv()
        res = simulate_plant(name, ctrl, env, duration=6.0, target_func=step_target, device=device)
        step_res[name] = res
        
        # Calculate metrics
        t = res['time']
        x = res['x']
        theta = res['theta']
        post_idx = np.where(t >= 1.0)[0]
        
        peak_x = x[post_idx].max()
        overshoot = max(0.0, (peak_x - 0.15) / 0.15) * 100.0
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        iae = float(trapz_fn(np.abs(x[post_idx] - 0.15), t[post_idx]))
        
        # Settling time within 5% band (0.15 +/- 0.0075 m)
        band = 0.05 * 0.15
        ts = 5.0
        for i in range(len(post_idx) - 200):
            idx = post_idx[i]
            if np.all(np.abs(x[idx:] - 0.15) <= band):
                ts = t[idx] - 1.0
                break
                
        ess = abs(x[-1] - 0.15) * 1000.0 # mm
        print(f"[{name:17s}] Overshoot: {overshoot:5.2f}% | IAE: {iae:6.4f} m·s | Settling: {ts:4.2f} s | ess: {ess:5.2f} mm")

    # -----------------------------------------------------------------
    # Test 2: External Lateral Disturbance Kick (0.15 N for 50 ms at t=3.0s)
    # -----------------------------------------------------------------
    print("\n--- Scenario 2: External Disturbance Rejection (0.15 N at t = 3.0 s) ---")
    def dist_force(t):
        return 0.15 if (3.0 <= t < 3.05) else 0.0

    for name, (col, ls, ctrl) in controllers.items():
        env = BallPlaneEnv()
        res = simulate_plant(name, ctrl, env, duration=6.0, target_func=step_target, dist_func=dist_force, device=device)
        post_dist = np.where(res['time'] >= 3.0)[0]
        max_defl = float(np.abs(res['x'][post_dist] - 0.15).max()) * 1000.0 # mm
        print(f"[{name:17s}] Max Disturbance Deflection: {max_defl:5.2f} mm")

    # -----------------------------------------------------------------
    # Plot & Save Simulation Figure
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    time_arr = step_res['Connectome (279)']['time']
    
    # 1. Ball Position
    for name, (col, ls, _) in controllers.items():
        axes[0].plot(time_arr, step_res[name]['x'] * 100, label=name, color=col, linestyle=ls, lw=1.8)
    axes[0].plot(time_arr, step_res['Connectome (279)']['ref'] * 100, 'k--', label='Reference Target', alpha=0.7)
    axes[0].set_ylabel('Ball Position x [cm]', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')
    axes[0].set_title('Ball-on-a-Plane: Nominal 15 cm Step Tracking Response', fontweight='bold')

    # 2. Plane Tilt Angle
    for name, (col, ls, _) in controllers.items():
        axes[1].plot(time_arr, np.rad2deg(step_res[name]['theta']), label=name, color=col, linestyle=ls, lw=1.8)
    axes[1].set_ylabel('Tilt θ [deg]', fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. Commanded Tilt u
    for name, (col, ls, _) in controllers.items():
        axes[2].plot(time_arr, np.rad2deg(step_res[name]['u']), label=name, color=col, linestyle=ls, lw=1.8)
    axes[2].set_ylabel('Command u [deg]', fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_fig = 'outputs/ballplane_simulation.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\n[Done] Simulation plot saved to: {out_fig}\n")


if __name__ == '__main__':
    run_simulation()
