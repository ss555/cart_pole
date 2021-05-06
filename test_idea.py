import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import socket
import logging
from logging import info, basicConfig
from collections import deque
import time
var=True
speed=4.1241
logname='TCP_SAC_DEBUG1'
arr=[]
def callback(way):
	global speed
	speed+=1.14
	arr.append(speed)
# info(f'f{var}')
basicConfig(filename=logname,
			filemode='w',#'a' for append
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)
info(f'done{var},and{speed}')#time9.131431579589844e-05
start_time=time.time()
callback(1.23)
print(time.time()-start_time)