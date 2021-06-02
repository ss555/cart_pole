#%%
import os, sys, socket
import itertools
from datetime import datetime





############################################
############################################

    
log = "exp_bin10_alpha_decay_"

now = datetime.now()
dt_string = now.strftime("%m%d_%H%M%S")

logdir = '/home/robotfish/Project/cart_pole/q_learning/'+dt_string
os.makedirs(logdir, exist_ok=True) 
os.chdir(logdir)

command = ['nohup python ../Qlearning.py',
            "> "+'/home/robotfish/Project/cart_pole/q_learning/'+dt_string+"/"+log+".out &"
            ]
command = " ".join(command)
print(command)
os.system(command)


# %%
