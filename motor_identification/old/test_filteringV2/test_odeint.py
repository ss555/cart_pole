from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

b = 0.25
c = 5.0

y0 = [np.pi - 0.1, 0.0]

t  = np.linspace(0, 10, 201)
# sol = odeint(pend, y0, t, args=(b, c))
sol=np.zeros(shape=(201,2))
sol[0]=y0
for i in range(len(t)-1):
    sol[i+1]=odeint(pend,y0,[0, 0.05],args=(b, c))[-1,:]
    y0      =sol[i+1]

plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()