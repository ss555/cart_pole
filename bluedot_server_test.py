from bluedot.btcomm import BluetoothServer
from signal import pause
import pandas as pd
import numpy as np
import io
import time

## when receive a data, send it back using the data_received function
def data_received(data_str):
    print(data_str+'\n')
    data = pd.read_csv(io.StringIO(data_str))
    time.sleep(20)
    s.send(data_str)
s = BluetoothServer(data_received)
pause()
