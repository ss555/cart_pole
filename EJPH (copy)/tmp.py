import numpy as np
import matplotlib.pyplot as plt
N = 100
x = np.arange(N)
fig = plt.figure()
ax = fig.add_subplot(111)
xx = x - (N/2.0)
plt.plot(xx, (xx*xx)-1225, label='$y=x^2$')
plt.plot(xx, 25*xx, label='$\sigma_\Theta$')
plt.plot(xx, -25*xx, label='$y=-25x$')
legend = plt.legend()
plt.show()