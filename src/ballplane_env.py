"""
Ball on a Plane (2D) Dynamical Simulation Environment.

Coordinates and State:
  x:      Ball position along the plane [m] (center/pivot at x = 0, x in [-0.30, +0.30])
  dx:     Ball linear velocity along the plane [m/s]
  theta:  Plane tilt angle relative to horizontal [rad] (in [-0.349, +0.349] rad = +/- 20 deg)

Control Input:
  u:      Commanded tilt angle [rad], clamped to [-theta_max, +theta_max]

"""

import numpy as np
import torch


class BallPlaneEnv:
    """
    Uses Runge-Kutta 4th order (RK4) integration with mechanical end stoppers.
    """
    def __init__(
        self,
        m=0.05,            # Ball mass [kg] (50 grams)
        r_b=0.015,         # Ball radius [m] (15 mm)
        g=9.81,            # Gravity [m/s^2]
        b=0.005,           # Rolling resistance / damping [N*s/m]
        L=0.60,            # Total plane length [m] (pivot at center)
        tau_m=0.05,        # Actuator / servo time constant [s] (50 ms)
        theta_max=0.34906585, # Maximum tilt angle [rad] (20 deg)
        dt=0.002           # Integration time-step [s] (2 ms, 500 Hz)
    ):
        self.m = m
        self.r_b = r_b
        self.g = g
        self.b = b
        self.L = L
        self.tau_m = tau_m
        self.theta_max = theta_max
        self.dt = dt
        
        self.x_min = -L / 2.0  # -0.30 m
        self.x_max =  L / 2.0  # +0.30 m
        
        # State: [x, dx, theta]
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
    def reset(self, x=0.0, dx=0.0, theta=0.0):
        self.state = np.array([float(x), float(dx), float(theta)], dtype=np.float64)
        return self.state.copy()
        
    def derivatives(self, state, u, dist_force=0.0):
        x = state[0]
        dx = state[1]
        theta = state[2]
        
        # Servo actuator dynamics: d(theta)/dt = (u - theta) / tau_m
        dtheta = (u - theta) / self.tau_m
        
        # Ball translational acceleration:
        # d2x = (5/7) * (g*sin(theta) - x*(dtheta^2)) - (5*b/(7*m))*dx + (5/(7*m))*dist_force
        c_centrif = - x * (dtheta ** 2)
        d2x = (5.0 / 7.0) * (self.g * np.sin(theta) + c_centrif) - (5.0 * self.b / (7.0 * self.m)) * dx + (5.0 / (7.0 * self.m)) * dist_force
        
        # Stopper constraints
        if x <= self.x_min and dx < 0:
            dx = 0.0
            d2x = max(0.0, d2x)
        elif x >= self.x_max and dx > 0:
            dx = 0.0
            d2x = min(0.0, d2x)
            
        return np.array([dx, d2x, dtheta], dtype=np.float64)
        
    def step_rk4(self, u, dist_force=0.0):
        # Physical actuator saturation
        u_clamped = np.clip(u, -self.theta_max, self.theta_max)
        
        dt = self.dt
        k1 = self.derivatives(self.state, u_clamped, dist_force)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, u_clamped, dist_force)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, u_clamped, dist_force)
        k4 = self.derivatives(self.state + dt * k3, u_clamped, dist_force)
        
        new_state = self.state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Stopper collision and boundary handling
        if new_state[0] <= self.x_min:
            new_state[0] = self.x_min
            if new_state[1] < 0:
                new_state[1] = 0.0
        elif new_state[0] >= self.x_max:
            new_state[0] = self.x_max
            if new_state[1] > 0:
                new_state[1] = 0.0
                
        # Servo tilt limit
        new_state[2] = np.clip(new_state[2], -self.theta_max, self.theta_max)
        
        self.state = new_state
        return self.state.copy()

    def step(self, u, dist_force=0.0, F_dist=0.0):
        force = dist_force if dist_force != 0.0 else F_dist
        return self.step_rk4(u, dist_force=force)


def ballplane_step_torch_rk4(
    state,
    u,
    dt=0.002,
    m=0.05,
    g=9.81,
    b=0.005,
    tau_m=0.05,
    theta_max=0.34906585,
    x_min=-0.30,
    x_max=0.30
):
    """
    Differentiable batched RK4 numerical step for PyTorch BPTT training.
    """
    u_clamped = torch.clamp(u, min=-theta_max, max=theta_max)
    
    def dynamics(s):
        x = s[:, 0]
        dx = s[:, 1]
        theta = s[:, 2]
        
        dtheta = (u_clamped - theta) / tau_m
        c_centrif = x * (dtheta ** 2)
        d2x = (5.0 / 7.0) * (g * torch.sin(theta) + c_centrif) - (5.0 * b / (7.0 * m)) * dx
        
        return torch.stack([dx, d2x, dtheta], dim=-1)
        
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)
    
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    # Soft limits during training for gradient preservation
    x_clamped = torch.clamp(next_state[:, 0], min=x_min, max=x_max)
    dx_clamped = next_state[:, 1]
    theta_clamped = torch.clamp(next_state[:, 2], min=-theta_max, max=theta_max)
    
    return torch.stack([x_clamped, dx_clamped, theta_clamped], dim=-1)
