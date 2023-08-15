import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravity
l = 1.0  # pendulum length
m = 1.0  # pendulum mass

# Simulation parameters
T = 10  # total time to simulate
dt = 0.01  # time step
N = int(T / dt)  # number of time steps
t = np.linspace(0, T, N)  # time grid


# System dynamics
def dynamics(theta, omega, u):
    return omega, -g / l * np.sin(theta) + u / (m * l * l)


# Lyapunov-based control law
def control_law(theta, omega):
    # Parameters of the control law
    K = np.array([10, 2])

    # Desired state
    theta_d = np.pi
    omega_d = 0

    # Control law
    u = -K @ np.array([theta - theta_d, omega - omega_d])
    return u


# Initialize state
theta = np.zeros(N)
omega = np.zeros(N)
theta[0] = 0
omega[0] = 0

# Simulate the pendulum
for i in range(N - 1):
    u = control_law(theta[i], omega[i])
    dtheta, domega = dynamics(theta[i], omega[i], u)
    theta[i + 1] = theta[i] + dt * dtheta
    omega[i + 1] = omega[i] + dt * domega

# Plot results
plt.figure()
plt.plot(t, theta, label='theta')
plt.plot(t, omega, label='omega')
plt.legend()
plt.show()
