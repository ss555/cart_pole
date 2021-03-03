import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
filename='./Transfer_learning/TCP_SAC_DEBUG'
data=[]
# 11:09:05,576 root INFO state: [ 0.0198      0.35494065 -0.99912286  0.04187565 -1.0422164 ], cost8.514195680618284e-05, done:False
with open(filename) as f:
    data = f.read()
    state = []
    pattern = re.compile(r'\[([^\]]+)[^d]+([\d\.]+)')
    # pattern = re.compile(r'\[([\d\b\.\-])+')
    # matches = pattern.sub(r'\1', line)
    matches = pattern.finditer(data)
    for mat in matches:
        state.append(mat[0])
    # reader = csv.reader(f)
    # database = pd.read_table(filename)
    # # print(database)
    # for row in reader:
    #     data-old.append(list(row))
    # print(data-old)
