%% Parameters for Ball on a Plane (2D) System
% Physical model of a solid ball rolling without slipping on a tilting plane.
%
% Coordinates:
%   x:      Ball position along the plane [m] (x = 0 is center/pivot)
%   dx:     Ball velocity along the plane [m/s]
%   theta:  Tilt angle of the plane relative to horizontal [rad]
%           (theta > 0 tilts plane downward to the right, accelerating ball in +x)
%   u:      Commanded tilt angle [rad] (from controller / actuator)


%% Physical Parameters
m       = 0.05;         % Mass of the rolling ball [kg] (50 grams)
r_b     = 0.015;        % Radius of the ball [m] (15 mm)
g       = 9.81;         % Acceleration due to gravity [m/s^2]
J_b     = (2/5)*m*r_b^2;% Ball moment of inertia (solid sphere) [kg*m^2]
b       = 0.005;        % Viscous rolling resistance / damping [N*s/m]
L       = 0.60;         % Length of the plane [m] (pivot at center)
tau_m   = 0.05;         % Actuator / servo time constant [s] (50 ms)

%% Operational & Physical Constraints
theta_max = 20 * pi / 180;  % Maximum plane tilt angle [rad] (~0.349 rad = 20 deg)
x_min     = -L / 2;         % Left end stopper of the plane [m] (-0.30 m)
x_max     =  L / 2;         % Right end stopper of the plane [m] (+0.30 m)

%% Initial Conditions
x_init      = 0.0;      % Initial ball position [m] (at center)
dx_init     = 0.0;      % Initial ball velocity [m/s]
theta_init  = 0.0;      % Initial plane angle [rad] (horizontal)