"""
Differentiable Physics Training for Inverted Cart-Pendulum Controllers.
Trains neural controllers via Backpropagation Through Time (BPTT) through
the Runge-Kutta 4th order (RK4) non-linear dynamics.

Usage:
    python3 train_cart_pendulum.py --model connectome --epochs 300
    python3 train_cart_pendulum.py --model all --epochs 300
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

from src.cart_pole_env import CartPendulumEnv
from src.models import (
    CElegansFullConnectomeController,
    MLPController,
    LSTMController
)


def train_cart_pendulum_model(
    model,
    num_epochs=300,
    batch_size=32,
    horizon=180,
    lr=0.01,
    weight_decay=1e-4,
    device='cpu',
    name='Model',
    save_path=None
):
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    env = CartPendulumEnv()
    
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    print(f"\n==================================================")
    print(f"[{name}] Starting BPTT training: {num_epochs} epochs (horizon={horizon}, batch={batch_size}, lr={lr})")
    print(f"==================================================")
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        # Sample initial states around upright:
        # theta in [-0.175, +0.175] rad (~ +/- 10 deg), x in [-0.4, 0.4] m
        x0 = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.8
        x_dot0 = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.2
        th0 = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.35
        th_dot0 = (torch.rand(batch_size, 1, device=device) - 0.5) * 0.2
        state = torch.cat([x0, x_dot0, th0, th_dot0], dim=-1)
        
        hidden = model.init_hidden(batch_size, device=device)
        total_loss = 0.0
        prev_action = torch.zeros(batch_size, device=device)
        
        for t in range(horizon):
            action, hidden = model.forward_step(state, hidden)
            if action.dim() > 1:
                action = action.squeeze(-1)
                
            next_state = env.step_torch_rk4(state, action)
            
            x = next_state[:, 0]
            x_dot = next_state[:, 1]
            th = next_state[:, 2]
            th_dot = next_state[:, 3]
            
            # Action smoothness penalty prevents high-frequency chatter
            act_diff = action - prev_action
            
            step_loss = (
                25.0 * (th ** 2) +
                2.5 * (th_dot ** 2) +
                4.0 * (x ** 2) +
                0.5 * (x_dot ** 2) +
                0.01 * (action ** 2) +
                0.08 * (act_diff ** 2)
            ).mean()
            
            total_loss = total_loss + step_loss
            prev_action = action
            state = next_state
            
        avg_loss = total_loss / horizon
        avg_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
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
    parser = argparse.ArgumentParser(description="Train Cart-Pendulum neural controllers using differentiable physics.")
    parser.add_argument("--model", type=str, default="connectome", choices=["connectome", "mlp", "lstm", "all"],
                        help="Which model to train (connectome, mlp, lstm, or all). Default: connectome.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs (default: 300).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--horizon", type=int, default=180, help="Simulation trajectory steps (default: 180).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01).")
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
            model = CElegansFullConnectomeController(data_dir=data_dir, u_max=20.0)
            save_path = os.path.join(args.save_dir, "c_elegans_full_connectome.pt")
            train_cart_pendulum_model(
                model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                horizon=args.horizon,
                lr=args.lr,
                device=device,
                name="Connectome (279 neurons)",
                save_path=save_path
            )
        elif m_name == "mlp":
            model = MLPController(in_features=4, hidden_dims=[128, 96, 55], u_max=20.0)
            save_path = os.path.join(args.save_dir, "mlp_controller.pt")
            train_cart_pendulum_model(
                model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                horizon=args.horizon,
                lr=args.lr,
                device=device,
                name="MLP (279 hidden)",
                save_path=save_path
            )
        elif m_name == "lstm":
            model = LSTMController(in_features=4, hidden_size=64, u_max=20.0)
            save_path = os.path.join(args.save_dir, "lstm_controller.pt")
            train_cart_pendulum_model(
                model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                horizon=args.horizon,
                lr=args.lr,
                device=device,
                name="LSTM (H=64)",
                save_path=save_path
            )


if __name__ == "__main__":
    main()
