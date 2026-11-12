This standalone directory contains files required to simulate, benchmark, and train biologically wired *C. elegans* somatic connectome controllers alongside scale-matched Multi-Layer Perceptrons (MLP), Long Short-Term Memory (LSTM) networks, and Linear Quadratic Regulators (LQR) across two dynamical benchmark systems:

1. **Inverted Cart-Pendulum** (underactuated, non-linear inverted pendulum on a movable cart).
2. **Ball-on-a-Plane** (2D underactuated ball rolling on an actuated tilting plane).


---

## 📁 Directory Structure

```text
elegans_simulation/
├── data/
│   └── connectome/
│       ├── Celegans_Varshney.csv       # Biological synaptic connectivity matrix (Varshney et al., 2011)
│       ├── all_cell_info.csv           # Neuron classifications (sensory, interneuron, motor)
│       └── herm_full_edgelist.csv      # Complete chemical and electrical edge list
├── models/                             # Pre-trained PyTorch checkpoints
│   ├── c_elegans_full_connectome.pt   # Cart-pendulum connectome controller
│   ├── mlp_controller.pt              # Cart-pendulum scale-matched MLP
│   ├── lstm_controller.pt             # Cart-pendulum LSTM
│   ├── connectome_ballplane.pt        # Ball-on-plane connectome controller
│   ├── mlp_ballplane.pt               # Ball-on-plane scale-matched MLP
│   └── lstm_ballplane.pt              # Ball-on-plane LSTM
├── outputs/                            # Simulation trajectory plots (.png)
├── simulink_models/                    # Reference MATLAB / Simulink plant definitions
│   ├── cart_pendulum/
│   │   ├── cartPendulum.slx           # Simulink physical plant model
│   │   └── cartPendulum_param.m       # Physical parameters script (M, m, l, g)
│   └── ball_plane/
│       ├── ballPlane.slx              # Simulink physical plant model with servo lag
│       └── ballPlane_param.m          # Physical parameters script (m, r_b, L, tau_m, b)
├── src/
│   ├── ballplane_env.py               # Non-linear Ball-on-Plane environment (RK4, servo dynamics)
│   ├── cart_pole_env.py               # Non-linear Cart-Pendulum environment (RK4)
│   ├── connectome.py                  # Connectome graph builder & adjacency parser
│   └── models.py                      # Unified model definitions (Connectome, MLP, LSTM, LQR)
├── requirements.txt                   # Minimal Python dependencies
├── simulate_cart_pendulum.py          # Evaluates pre-trained models on Cart-Pendulum
├── simulate_ballplane.py              # Evaluates pre-trained models on Ball-on-a-Plane
├── train_cart_pendulum.py             # Trains Cart-Pendulum controllers via BPTT
└── train_ballplane.py                 # Trains Ball-on-a-Plane controllers via BPTT
```

---

## 🚀 Quick Start

### 1. Installation

Create a clean Python virtual environment and install the minimal dependencies:

```bash
cd elegans_simulation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Running Simulations (Pre-Trained Models)

Both simulation scripts load the saved `.pt` weights, execute the non-linear closed-loop scenarios, compute benchmark metrics (settling time, control effort, overshoot, disturbance deflection), and export high-resolution response plots to `outputs/`.

#### Inverted Cart-Pendulum
Evaluates severe tilt recovery ($\theta_0 = 11.46^\circ$) and external impulse kick ($+5.0\,\text{N}$ at $t = 2.0\,\text{s}$):
```bash
python3 simulate_cart_pendulum.py
```
*Output plot:* `outputs/cart_pendulum_simulation.png`

#### Ball-on-a-Plane
Evaluates a $15\,\text{cm}$ step trajectory and lateral disturbance rejection ($0.15\,\text{N}$ at $t = 3.0\,\text{s}$):
```bash
python3 simulate_ballplane.py
```
*Output plot:* `outputs/ballplane_simulation.png`

---

### 3. Training Controllers (Differentiable Physics BPTT)

The neural controllers are trained by backpropagating loss gradients directly through the non-linear 4th-order Runge-Kutta (RK4) dynamical physics equations.

#### Train Cart-Pendulum Controllers:
```bash
# Train the biologically wired Connectome controller (default: 300 epochs)
python3 train_cart_pendulum.py --model connectome --epochs 300

# Or train all architectures (connectome, mlp, lstm)
python3 train_cart_pendulum.py --model all --epochs 300
```

#### Train Ball-on-a-Plane Controllers:
```bash
# Train the biologically wired Connectome controller (default: 250 epochs)
python3 train_ballplane.py --model connectome --epochs 250

# Or train all architectures (connectome, mlp, lstm)
python3 train_ballplane.py --model all --epochs 250
```

#### Training Options:
- `--epochs INT`: Number of training epochs.
- `--batch-size INT`: Batch size (default: 32).
- `--horizon INT`: Number of rollout steps per trajectory.
- `--lr FLOAT`: Learning rate.
- `--save-dir PATH`: Directory where checkpoints are saved (default: `models`).
