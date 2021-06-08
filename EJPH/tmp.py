import numpy as np
# import matplotlib as mpl
#
# ## agg backend is used to create plot as a .png file
# mpl.use('agg')

import matplotlib.pyplot as plt




N_TRIALS=10
THETA_DOT_THRESHOLD=10
theta=np.linspace(-np.pi,np.pi,N_TRIALS)
theta_dot=np.linspace(-THETA_DOT_THRESHOLD,THETA_DOT_THRESHOLD,N_TRIALS)
arr=np.transpose([np.tile(theta, len(theta_dot)), np.repeat(theta_dot, len(theta))])
plt.plot(arr[:,0],arr[:,1],'.')
plt.show()
