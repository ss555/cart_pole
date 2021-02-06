from bluedot.btcomm import BluetoothClient
from signal import pause
import numpy as np
import pandas as pd
import time

def data_received(data):
    print('data received from the server')
    print(data)
    global state
    state = "continue"

c = BluetoothClient("sardor-Precision-7550", data_received)#C0:3E:BA:6C:9E:EB sc
state = 'continue'
while True:
    if state == 'continue':
        data = np.random.rand(3,2)
        df = pd.DataFrame(data)
        data_str = df.to_string()
        state = 'stop'
        c.send(data_str)
    else:
        time.sleep(1)
