"""
Differentiable Physics Training for Ball-on-a-Plane Controllers.
Trains neural controllers via Backpropagation Through Time (BPTT) through
the 4th-order Runge-Kutta (RK4) non-linear dynamics including servo lag.

Usage:
    python3 train_ballplane.py --model connectome --epochs 250
    python3 train_ballplane.py --model all --epochs 250
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ballplane_env import BallPlaneEnv, ballplane_step_torch_rk4
from src.models import (
    BallPlaneConnectomeController,
    BallPlaneMLPController,
    BallPlaneLSTMController
)


def train_ballplane_model(
    model,
    num_epochs=250,
    batch_size=32,
    horizon=300,
    dt=0.002,
    lr=0.005,
    weight_decay=1e-4,
    device='cpu',
    name='Model',
    save_path=None
):
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    
    theta_max = 0.34906585
    x_scale = 0.15
    v_scale = 0.50
    
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    print(f"\n==================================================")
    print(f"[{name}] Starting BPTT training: {num_epochs} epochs (horizon={horizon}, dt={dt*1000:.1f}ms, batch={batch_size}, lr={lr})")
    print(f"==================================================")
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        # Sample randomized initial states
        # Ball position in [-0.20, +0.20] m
        x_init = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.40
        dx_init = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.20
        theta_init = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.20
        state = torch.cat([x_init, dx_init, theta_init], dim=-1)
        
        # Sample randomized target position in [-0.20, +0.20] m
        x_ref = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.40
        
        hidden = model.init_hidden(batch_size, device=device)
        total_loss = 0.0
        prev_u = torch.zeros(batch_size, device=device)
        
        for t in range(horizon):
            x_curr = state[:, 0:1]
            dx_curr = state[:, 1:2]
            theta_curr = state[:, 2:3]
            
            # Observation vector: [(x - x_ref)/x_scale, dx/v_scale, theta/theta_max, x_ref/x_scale]
            e_x = x_curr - x_ref
            obs = torch.cat([e_x / x_scale, dx_curr / v_scale, theta_curr / theta_max, x_ref / x_scale], dim=-1)
            
            u_cmd, hidden = model.forward_step(obs, hidden)
            if u_cmd.dim() > 1:
                u_cmd = u_cmd.squeeze(-1)
                
            next_state = ballplane_step_torch_rk4(state, u_cmd, dt=dt)
            
            x_next = next_state[:, 0]
            dx_next = next_state[:, 1]
            theta_next = next_state[:, 2]
            
            pos_err = (x_next - x_ref.squeeze(-1)) / x_scale
            vel_err = dx_next / v_scale
            th_err = theta_next / theta_max
            u_pen = u_cmd / theta_max
            u_diff = (u_cmd - prev_u) / theta_max
            
            # Stopper soft penalty
            stopper_penalty = torch.relu(-0.28 - x_next) ** 2 + torch.relu(x_next - 0.28) ** 2
            
            step_loss = (
                30.0 * (pos_err ** 2) +
                6.0 * (vel_err ** 2) +
                1.5 * (th_err ** 2) +
                0.05 * (u_pen ** 2) +
                0.10 * (u_diff ** 2) +
                100.0 * stopper_penalty
            ).mean()
            
            total_loss = total_loss + step_loss
            prev_u = u_cmd
            state = next_state
            
        avg_loss = total_loss / horizon
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = avg_loss.item()
        loss_history.append(loss_val)
        
        if loss_val < best_loss and save_path is not None:
            best_loss = loss_val
            torch.save(model.state_dict(), save_path)
            
        if epoch % 25 == 0 or epoch == 1 or epoch == num_epochs:
            elapsed = time.time() - start_time
            print(f"[{name}] Epoch {epoch:3d}/{num_epochs:3d} | Loss: {loss_val:.4f} (Best: {best_loss:.4f}) | Elapsed: {elapsed:.1f}s")
            
    print(f"[{name}] Training finished in {time.time() - start_time:.1f}s.")
    if save_path:
        print(f"[{name}] Best weights saved to: {save_path}")
    return loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Ball-on-a-Plane neural controllers using differentiable physics.")
    parser.add_argument("--model", type=str, default="connectome", choices=["connectome", "mlp", "lstm", "all"],
                        help="Which model to train (connectome, mlp, lstm, or all). Default: connectome.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs (default: 250).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--horizon", type=int, default=300, help="Simulation trajectory steps (default: 300).")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save checkpoints (default: models).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    to_train = ["connectome", "mlp", "lstm"] if args.model == "all" else [args.model]
    
    for m_name in to_train:
        if m_name == "connectome":
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "connectome")
            model = BallPlaneConnectomeController(data_dir=data_dir)
            save_path = os.path.join(args.save_dir, "connectome_ballplane.pt")
            train_ballplane_model(
                model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                horizon=args.horizon,
                lr=0.006 if args.lr == 0.005 else args.lr,
                device=device,
                name="Connectome (279 neurons)",
                save_path=save_path
            )
        elif m_name == "mlp":
            model = BallPlaneMLPController()
            save_path = os.path.join(args.save_dir, "mlp_ballplane.pt")
            train_ballplane_model(
                model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                horizon=args.horizon,
                lr=0.004 if args.lr == 0.005 else args.lr,
                device=device,
                name="MLP (279 hidden)",
                save_path=save_path
            )
        elif m_name == "lstm":
            model = BallPlaneLSTMController()
            save_path = os.path.join(args.save_dir, "lstm_ballplane.pt")
            train_ballplane_model(
                model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                horizon=args.horizon,
                lr=0.004 if args.lr == 0.005 else args.lr,
                device=device,
                name="LSTM (H=64)",
                save_path=save_path
            )


if __name__ == "__main__":
    main()
