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

start_time=time.time()
info(f'done{var},and{speed}')#time9.131431579589844e-05
print(time.time()-start_time)
# from env_wrappers import ResultsWriter
# res=ResultsWriter(
#                 'test',
#                 header={"t_start": 0, "env_id": 0 and 0}
#             )
#
#
# start_time=time.time()
# res.write_row({'r':5,'l':3})
# print(time.time()-start_time)


'''
#FIND ALL JUPYTER FILES
import os

# This is the path where you want to search
path = r'./../..'

# this is the extension you want to detect
extension = '.ipynb'

for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            print(file_name)
            print(file_name_path)   # This is the full path of the filter file'''