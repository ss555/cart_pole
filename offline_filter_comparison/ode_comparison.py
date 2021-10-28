import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint  # for comparison
from utils import rungekutta4
def pendulum(y, t, b, c):
    return np.array([y[1], -b*y[1] - c*np.sin(y[0])])

def semi_euler(f, y0, t, args=()):
    n=len(t)

    y=np.zeros((n,len(y0)))
    y[0]=y0
    for i in range(n-1):
        h = (t[i + 1] - t[i])
        dqdt = f(y[i],t[i])
        y[i+1,1] = y[i,1] + dqdt[1] * h
        y[i+1,0] = y[i,0] + y[i+1,1] * h
        y[i+1,3] = y[i,3] + dqdt[3] * h
        y[i+1,2] = y[i,2] + y[i+1,3] * h
    return y
def pend2(state, t, action=0, fa=0, fb=0, fc=0, masspole=0.07,g=9.81,length=0.43, kPendViscous = 0.07):
    [x, x_dot, theta, theta_dot] = state
    dqdt = np.zeros_like(state)
    costheta, sintheta = [np.cos(theta), np.sin(theta)]
    dqdt[0] = state[1]
    force = 0#self._calculate_force(action,cart_speed=dqdt[0])  # 0.44*(fa*x_dot+fb*self.tensionMax*action[0]+fc*np.sign(x_dot))
    dqdt[1] = (force + 0.07 * theta_dot ** 2 * length * sintheta + masspole * g * sintheta * costheta) / (0.5 + 0.07 * sintheta ** 2)
    dqdt[3] = -4.8 ** 2 * sintheta + dqdt[1] / 0.47 * costheta - theta_dot * kPendViscous
    dqdt[2] = state[3]
    return dqdt
def pend(state, t, action=0, fa=0, fb=0, fc=0, fd=0.07, masscart=0.45, masspole=0.07,g=9.81,length=0.43, kPendViscous = 0.07):
    [x, x_dot, theta, theta_dot] = state
    dqdt = np.zeros_like(state)
    costheta, sintheta = [np.cos(theta), np.sin(theta)]
    dqdt[0] = state[1]
    if action == 0:
        force = masscart * (fa * dqdt[0] + fc * np.sign(dqdt[0]))
    else:
        force = masscart * (fa * dqdt[0] + fb * (12 * action) + fc * np.sign(
            dqdt[0]) + fd)
    force=0
    dqdt[3] = (force + masspole * theta_dot ** 2 * length * sintheta + masspole * g * sintheta * costheta) / (
                          masscart + masspole * sintheta ** 2)
    dqdt[2] = state[3]
    return dqdt
# pendulum
b = 0.25
c = 5.0
y0 = np.array([np.pi - 0.1, 0.0])
t = np.linspace(0, 10, 101)
sol = odeint(pendulum, y0, t, args=(b, c))

# sol = rungekutta4(pend, y0, t, args=(b, c))
plt.plot(t, sol[:, 0], 'b', label=r'$\theta(t)$')
plt.plot(t, sol[:, 1], 'g', label=r'$\omega(t)$')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

# cartpole
t=np.linspace(0,10,int(10/0.05))
[x, x_dot, theta, theta_dot] = [0,0,np.pi-0.1,0]
arr = odeint(pend2,y0=[x, x_dot, theta, theta_dot],t=t,args=())
arr2 = rungekutta4(pend2,y0=[x, x_dot, theta, theta_dot],t=t,args=())
Te=0.05
tSE=np.linspace(0,10,int(10/Te))
arr3 = semi_euler(pend2,y0=[x, x_dot, theta, theta_dot],t=tSE,args=())
plt.plot(t, arr[:, 2],  label=f'$\Theta(t)$ odeint Te={Te}')
plt.plot(t, arr2[:, 2], label=rf'$\theta(t)$ rk4 Te={Te}')
plt.plot(tSE, arr3[:, 2], label=rf'$\theta(t)$ semi-euler Te={Te}')
plt.plot(t, arr[:, 0], label=rf'$x(t)$ odeint Te={Te}')
plt.plot(t, arr2[:, 0], label=rf'$x(t)$ rk4 Te={Te}')
plt.plot(tSE, arr3[:, 0], label=rf'$x(t)$ semi-euler Te={Te}')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.savefig('ode_comparison')
plt.show()