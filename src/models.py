"""
Neural Network and Classical Controllers for:
1. Inverted Cart-Pendulum
2. Ball-on-a-Plane

Includes:
- Biological C. elegans Full Somatic Connectome Controller (279 neurons, 97.97% sparse)
- Scale-Matched Multi-Layer Perceptron (MLP) Controller
- Long Short-Term Memory (LSTM) Controller
- Saturated Linear Quadratic Regulator (LQR / LQI) Controller
"""

import os
import sys
import numpy as np
import scipy.linalg
from scipy.signal import cont2discrete
import torch
import torch.nn as nn

# Local connectome loader
from src.connectome import build_full_connectome

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "connectome"))


# =============================================================================
# 1. CART-PENDULUM CONTROLLERS
# =============================================================================

class CElegansFullConnectomeController(nn.Module):
    """
    Full 279-neuron C. elegans Somatic Connectome Controller for Cart-Pendulum.
    """
    def __init__(self, data_dir=DEFAULT_DATA_DIR, u_max=20.0, dt=0.02):
        super().__init__()
        fc = build_full_connectome(data_dir)
        self.n_neurons = fc['n_neurons']
        self.u_max = u_max
        self.dt = dt
        
        mask = (fc['adj_matrix'] > 0).astype(np.float32)
        np.fill_diagonal(mask, 1.0)
        self.register_buffer('adj_mask', torch.tensor(mask, dtype=torch.float32))
        
        self.W = nn.Parameter(torch.randn(self.n_neurons, self.n_neurons) * 0.15)
        self.tau = nn.Parameter(torch.ones(self.n_neurons) * 0.05)
        self.v_leak = nn.Parameter(torch.zeros(self.n_neurons))
        
        self.sens_in = nn.Linear(4, self.n_neurons, bias=False)
        self.motor_out = nn.Linear(self.n_neurons, 1, bias=False)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = self.W.device
        return torch.zeros(batch_size, self.n_neurons, device=device)

    def forward_step(self, obs, v_prev):
        effective_w = self.W * self.adj_mask
        synaptic_input = torch.matmul(torch.tanh(v_prev), effective_w.t())
        sensory_input = self.sens_in(obs)
        
        tau_clamped = torch.clamp(torch.abs(self.tau), min=0.005, max=0.5)
        dv = (-(v_prev - self.v_leak) + synaptic_input + sensory_input) / tau_clamped
        v_next = torch.clamp(v_prev + self.dt * dv, -5.0, 5.0)
        
        action = self.u_max * torch.tanh(self.motor_out(torch.tanh(v_next))).squeeze(-1)
        return action, v_next

    def get_sparsity(self):
        total = self.adj_mask.numel()
        active = torch.count_nonzero(self.adj_mask).item()
        return 1.0 - (active / total)


class MLPController(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Controller for Cart-Pendulum.
    128 -> 96 -> 55
    """
    def __init__(self, in_features=4, hidden_dims=[128, 96, 55], u_max=20.0):
        super().__init__()
        self.u_max = u_max
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim, bias=False))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1, bias=False))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def init_hidden(self, batch_size=1, device=None):
        return None

    def forward_step(self, obs, hidden=None):
        raw_act = self.net(obs).squeeze(-1)
        return self.u_max * raw_act, None


class LSTMController(nn.Module):
    """
    Long Short-Term Memory (LSTM) Recurrent Controller for Cart-Pendulum.
    Hidden size H=64.
    """
    def __init__(self, in_features=4, hidden_size=64, u_max=20.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.u_max = u_max
        self.lstm_cell = nn.LSTMCell(in_features, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward_step(self, obs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(obs.shape[0], device=obs.device)
        h, c = self.lstm_cell(obs, hidden)
        action = self.u_max * torch.tanh(self.readout(h)).squeeze(-1)
        return action, (h, c)

    def get_sparsity(self):
        return 0.0


class CartPendulumLQRController:
    """Discrete Linear Quadratic Regulator (LQR) with saturation for Cart-Pendulum."""
    def __init__(self, M=1.0, m=0.5, l=0.5, g=9.81, dt=0.02, u_max=20.0):
        self.u_max = u_max
        Ac = np.array([
            [0, 1, 0, 0],
            [0, 0, m * g / M, 0],
            [0, 0, 0, 1],
            [0, 0, (M + m) * g / (M * l), 0]
        ], dtype=np.float64)
        Bc = np.array([
            [0],
            [1.0 / M],
            [0],
            [1.0 / (M * l)]
        ], dtype=np.float64)
        Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(4), np.zeros((4, 1))), dt)
        Q = np.diag([4.0, 0.5, 25.0, 2.5])
        R_mat = np.array([[0.01]])
        P_dare = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R_mat)
        self.K = np.linalg.inv(R_mat + Bd.T @ P_dare @ Bd) @ (Bd.T @ P_dare @ Ad)

    def init_hidden(self, batch_size=1, device=None):
        return None

    def compute_action(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        u = float(-self.K @ obs)
        return float(np.clip(u, -self.u_max, self.u_max))


# =============================================================================
# 2. BALL-ON-A-PLANE CONTROLLERS
# =============================================================================

class BallPlaneConnectomeController(nn.Module):
    """
    Biological C. elegans Somatic Connectome Controller for Ball-on-a-Plane.
    """
    def __init__(self, data_dir=DEFAULT_DATA_DIR, theta_max=0.34906585, dt=0.002):
        super().__init__()
        fc = build_full_connectome(data_dir)
        self.n_neurons = fc['n_neurons']
        self.theta_max = theta_max
        self.dt = dt
        
        mask = (fc['adj_matrix'] > 0).astype(np.float32)
        np.fill_diagonal(mask, 1.0)
        self.register_buffer('adj_mask', torch.tensor(mask, dtype=torch.float32))
        
        self.W = nn.Parameter(torch.randn(self.n_neurons, self.n_neurons) * 0.15)
        self.tau = nn.Parameter(torch.ones(self.n_neurons) * 0.05)
        self.v_leak = nn.Parameter(torch.zeros(self.n_neurons))
        
        self.sens_in = nn.Linear(4, self.n_neurons, bias=False)
        self.motor_out = nn.Linear(self.n_neurons, 1, bias=False)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = self.W.device
        return torch.zeros(batch_size, self.n_neurons, device=device)

    def forward_step(self, obs, v_prev):
        effective_w = self.W * self.adj_mask
        synaptic_input = torch.matmul(torch.tanh(v_prev), effective_w.t())
        sensory_input = self.sens_in(obs)
        
        tau_clamped = torch.clamp(torch.abs(self.tau), min=0.005, max=0.5)
        dv = (-(v_prev - self.v_leak) + synaptic_input + sensory_input) / tau_clamped
        v_next = torch.clamp(v_prev + self.dt * dv, -5.0, 5.0)
        
        raw_motor = self.motor_out(torch.tanh(v_next)).squeeze(-1)
        action = self.theta_max * torch.tanh(raw_motor)
        return action, v_next


class BallPlaneMLPController(nn.Module):
    """Multi-Layer Perceptron for Ball-on-a-Plane."""
    def __init__(self, in_features=4, hidden_dims=[128, 96, 55], theta_max=0.34906585):
        super().__init__()
        self.theta_max = theta_max
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim, bias=True))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1, bias=True))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def init_hidden(self, batch_size=1, device=None):
        return None

    def forward_step(self, obs, hidden=None):
        raw_act = self.net(obs).squeeze(-1)
        return self.theta_max * raw_act, None


class BallPlaneLSTMController(nn.Module):
    """Long Short-Term Memory Controller for Ball-on-a-Plane."""
    def __init__(self, in_features=4, hidden_size=64, theta_max=0.34906585):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.theta_max = theta_max
        self.lstm_cell = nn.LSTMCell(in_features, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward_step(self, obs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(obs.shape[0], device=obs.device)
        h, c = self.lstm_cell(obs, hidden)
        u = self.theta_max * torch.tanh(self.readout(h)).squeeze(-1)
        return u, (h, c)

    def get_sparsity(self):
        return 0.0


class BallPlaneLQRController:
    """Discrete Linear Quadratic Regulator with Integral Action (LQI) for Ball-on-a-Plane."""
    def __init__(self, g=9.81, tau_m=0.05, dt=0.002, theta_max=0.34906585):
        self.dt = dt
        self.theta_max = theta_max
        self.integral_error = 0.0
        
        # State: [x, dx, theta, int_e]
        Ac = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, (5.0 / 7.0) * g, 0.0],
            [0.0, 0.0, -1.0 / tau_m, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])
        Bc = np.array([
            [0.0],
            [0.0],
            [1.0 / tau_m],
            [0.0]
        ])
        Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(4), np.zeros((4, 1))), dt)
        Q = np.diag([80.0, 5.0, 1.0, 40.0])
        R = np.array([[0.5]])
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        self.K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    def reset(self):
        self.integral_error = 0.0

    def init_hidden(self, batch_size=1, device=None):
        return None

    def compute_action(self, x, dx, theta, x_ref):
        e_x = x - x_ref
        self.integral_error += e_x * self.dt
        self.integral_error = np.clip(self.integral_error, -0.5, 0.5)
        z = np.array([e_x, dx, theta, self.integral_error])
        u = float(-self.K @ z)
        return float(np.clip(u, -self.theta_max, self.theta_max))
