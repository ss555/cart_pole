#%%
import os, sys, socket
import itertools
from datetime import datetime





############################################
############################################

    
log = "exp_bin10_alpha_decay_"

now = datetime.now()
dt_string = now.strftime("%m%d_%H%M%S")
if 'Mac' in socket.gethostname():
    logdir = '/Users/lfu/GDrive/IBRID/5_Results/0_RL_fish/'+dt_string
elif 'pop' in socket.gethostname():
    logdir = '/home/robotfish/Project/cart_pole/Sarsa/'+dt_string
os.makedirs(logdir, exist_ok=True) 
os.chdir(logdir)

command = ['nohup /home/robotfish/miniconda3/envs/ai/bin/python ../sarsa.py',
            "> "+'/home/robotfish/Project/cart_pole/Sarsa/'+dt_string+"/"+log+".out &"
            ]

command = " ".join(command)
print(command)
os.system(command)


# %%
