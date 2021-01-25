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

c = BluetoothClient("B8:27:EB:6F:EE:40", data_received)
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
