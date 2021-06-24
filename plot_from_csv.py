import numpy as np
from matplotlib import pyplot as plt
data=np.genfromtxt('/home/pi/Project/angle_iden.csv',delimiter=',')
pos=data[2,:]/180
time=data[1,:]
plt.plot(time,pos)
plt.show()
