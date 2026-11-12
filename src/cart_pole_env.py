"""
Cart-Pendulum Environment
M = 1.0 kg (cart mass)
m = 0.5 kg (pendulum mass)
l = 0.5 m (pendulum length)
g = 9.81 m/s^2 (gravity)

State: [x, x_dot, theta, theta_dot]
Control input: u (force on cart [N])
"""

import numpy as np
import torch


class CartPendulumDynamics:
    def __init__(self, M=1.0, m=0.5, l=0.5, g=9.81):
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def derivatives_np(self, state, u):
        x, x_dot, theta, theta_dot = state
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        denom = self.M + self.m - self.m * (cos_th ** 2)
        
        x_ddot = (u - self.m * self.l * (theta_dot ** 2) * sin_th + self.m * self.g * sin_th * cos_th) / denom
        theta_ddot = (u * cos_th + (self.M + self.m) * self.g * sin_th - self.m * self.l * (theta_dot ** 2) * sin_th * cos_th) / (self.l * denom)
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float32)

    def derivatives_torch(self, state, u):
        x = state[:, 0:1]
        x_dot = state[:, 1:2]
        theta = state[:, 2:3]
        theta_dot = state[:, 3:4]
        if u.dim() == 1:
            u = u.unsqueeze(1)
            
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)
        denom = self.M + self.m - self.m * (cos_th ** 2)
        
        x_ddot = (u - self.m * self.l * (theta_dot ** 2) * sin_th + self.m * self.g * sin_th * cos_th) / denom
        theta_ddot = (u * cos_th + (self.M + self.m) * self.g * sin_th - self.m * self.l * (theta_dot ** 2) * sin_th * cos_th) / (self.l * denom)
        
        return torch.cat([x_dot, x_ddot, theta_dot, theta_ddot], dim=-1)


class CartPendulumEnv:
    def __init__(self, dt=0.02, max_steps=500, max_force=20.0, M=1.0, m=0.5, l=0.5, g=9.81):
        self.dt = dt
        self.max_steps = max_steps
        self.max_force = max_force
        self.dynamics = CartPendulumDynamics(M=M, m=m, l=l, g=g)
        self.state = None
        self.steps = 0

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float32).copy()
        else:
            # Random perturb around upright: theta in [-0.15, 0.15] rad (~ +/- 8.6 deg)
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),    # x
                np.random.uniform(-0.05, 0.05),  # x_dot
                np.random.uniform(-0.15, 0.15),  # theta
                np.random.uniform(-0.1, 0.1)     # theta_dot
            ], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        u = np.clip(float(action), -self.max_force, self.max_force)
        
        # 4th-order Runge-Kutta integration
        s = self.state
        k1 = self.dynamics.derivatives_np(s, u)
        k2 = self.dynamics.derivatives_np(s + 0.5 * self.dt * k1, u)
        k3 = self.dynamics.derivatives_np(s + 0.5 * self.dt * k2, u)
        k4 = self.dynamics.derivatives_np(s + self.dt * k3, u)
        self.state = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        self.steps += 1
        x, x_dot, theta, theta_dot = self.state
        
        # Cost / Reward
        # We penalize angle deviations heavily, cart position, and control effort
        cost = (
            5.0 * (theta ** 2) +
            0.5 * (theta_dot ** 2) +
            1.0 * (x ** 2) +
            0.1 * (x_dot ** 2) +
            0.001 * (u ** 2)
        )
        reward = -cost
        
        # Termination conditions (fall if angle > ~35 degrees or cart drifts off track > 2.5m)
        terminated = bool(abs(theta) > 0.6 or abs(x) > 2.5)
        truncated = bool(self.steps >= self.max_steps)
        
        return self.state.copy(), reward, terminated, truncated, {'u': u}

    def step_torch_rk4(self, state, action):
        """Vectorized differentiable RK4 step for PyTorch."""
        u = torch.clamp(action, -self.max_force, self.max_force)
        k1 = self.dynamics.derivatives_torch(state, u)
        k2 = self.dynamics.derivatives_torch(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics.derivatives_torch(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics.derivatives_torch(state + self.dt * k3, u)
        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state
