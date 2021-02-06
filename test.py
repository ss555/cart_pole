'''
		iFinCourseMoteur = pi.read(18);	
		iFinCourseEncodeur = pi.read(17);  
		
		if not iFinCourseMoteur and oldIFinCourseMoteur:	
			print('17')
			oldIFinCourseMoteur = iFinCourseMoteur
		else:
			oldIFinCourseMoteur = iFinCourseMoteur
		if not oldIFinCourseEncodeur and iFinCourseEncodeur:
			print('18')  
			oldIFinCourseEncodeur = iFinCourseEncodeur
		else:
			oldIFinCourseEncodeur = iFinCourseEncodeur

#!/usr/bin/env python3

# Take an user input
name = input("What is your name? ")
# Check the input value

if(name != ""):
   # Print welcome message if the value is not empty
   print("Hello %s, welcome to our site" %name )
else:
   # Print empty message
   print("The name can't be empty.")

# Wait for the user input to terminate the program
input("Press any key to terminate the program")
# Print bye message
print("See you later.")

import torch
import stable-baselines3
import sys
import time
import logging
print("Waiting for five seconds for calculating ...")
# Wait for 2 seconds
print(sys.version)
print("W.")
logname='DEBUG'
logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logging.info("Running Urban Planning")
'''
import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

env = gym.make('Pendulum-v0')

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
