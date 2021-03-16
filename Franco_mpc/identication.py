# Bunch of imports
import sys
import os
import argparse
import csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('./'))
# configure plots to be smaller
plt.rcParams['figure.figsize'] = [7, 4]
plt.rcParams['figure.dpi'] = 100

# Auxiliary function to select a file using a Qt5 dialog
def getDataFile():
  try:
    from PyQt5.QtWidgets import QApplication, QFileDialog
    qapp = QApplication([])
    file_name, _ = QFileDialog.getOpenFileName(caption="Select csv", filter="CSV Files (*.csv);;All Files (*)")
    return file_name
  except ModuleNotFoundError:
    print("WARNING: PyQt5 seems not to be installed. Files cannot be selected using QFileDialog.")
    return None


# Reads a CSV file and returns a dictionary whose keys are the header entries.
# Optionally, a selected number of fields can be specified.
# You can also ask the result to be returned as a plain tuple rather than a dictionary.
#
# EXAMPLES
#
# 1) Load everything as a dictionary:
#       data = loadCsv('my_data.csv')
# 'data' will be like: {'field1': np.array(...), 'field2': np.array(...), ...}
#
# 2) Load only the given fields:
#       data = loadCsv('my_data.csv', 'field8', 'field5')
# 'data' will cointain {'field8': np.array(...), 'field5': np.array(...)}
#
# 3) Do not create a dictionary:
#       f8,f5 = loadCsv('my_data.csv', 'field8', 'field5', as_dict=False)
# This is equivalent to the following:
#       data = loadCsv('my_data.csv', 'field8', 'field5')
#       f8 = data.get('filed8')
#       f5 = data.get('filed5')
#
def loadCsv(file_name, *fields, as_dict=True):
  with open(file_name) as f:
    # read just the header
    header = f.readline().strip().replace(" ", "").split(",")
    idx = []
    if len(fields) == 0:
      fields = header
      idx = range(len(header))
    else:
      for elem in fields:
        if elem not in header:
          raise ValueError(f"Field '{elem}' is not part of the header of the given file '{file_name}'. The header is: '{','.join(header)}'")
        idx.append(header.index(elem))
    # read again the file, using numpy
    data = np.loadtxt(f, delimiter=",")
    if as_dict:
      return {header[i]: data[:,i] for i in idx}
    else:
      return (data[:,i] for i in idx)

# with open(csv_file) as f:
#     # read just the header
#     header = f.readline().strip().replace(" ", "").split(",")

# Creates a set of subdictionaries from a dictionary created using loadCsv.
# This function is better explained via an example:
# Assume that you have the following CSV, containing data from a set of
# tests, each one having a different 'test_id' (do not focus on the content
# of 'input' and 'output'):
#
#   test_id, input, output
#         1,    10,     20
#         1,    20,     30
#         1,    30,     40
#         2,    40,     50
#         3,    50,     60
#         3,    60,     70
#         2,    70,     80
#
# It would be processed by a "plain" call to loadCsv as the following dict:
#
#   data = {
#     'test_id': np.array([1,1,1,2,3,3,2]),
#     'input': np.array([10,20,30,40,50,60,70]),
#     'output': np.array([20,30,40,50,60,70,80])
#   }
#
# However, you might want to group data depending on the test id.
# To do that, you can call thus function as:
#
#   classify('test_id', data)
#
# The output will be another dictionary in the following form:
#
#   {
#     1: {
#       'test_id': np.array([1,1,1]),
#       'input': np.array([10,20,30]),
#       'output': np.array([20,30,40])
#     },
#     2: {
#       'test_id': np.array([2,2]),
#       'input': np.array([40,70]),
#       'output': np.array([50,80])
#     },
#     3: {
#       'test_id': np.array([3,3]),
#       'input': np.array([50,60]),
#       'output': np.array([60,70])
#     }
#   }
# Note that each sub-dictionary contains the only data related to a specific 'test_id' value.
def classify(key, data_dict):
  if key not in data_dict:
    raise ValueError(f"Key '{key}' is not contained in the input dictionary. Valid keys are '{data_dict.keys()}'")
  key_signal = data_dict.get(key)
  out = dict()
  for val in np.unique(data_dict.get(key)):
    idx = key_signal==val
    out[val] = dict()
    for k,v in data_dict.items():
      out[val][k] = v[idx]
  return out

# Measured dynamic parameters
g = 9.806 # gravity
M = 0.439 # mass of the base
m1 = 0.012 # mass of the plastic component that connects the base and the rod
m2 = 0.019 # mass of the rod
m3 = 0.031 # mass of the terminal weight
ell = 0.495 # length of the rod
mt = M + m1 + m2 + m3
mu = m2*ell/2 + m3*ell
# The missing parameters are:
# - Friction coefficients
# - Total inertia of the rod and weight
# - Gain of the motor

# select the file to be opened, either "statically" or using Qt5
csv_file = 'logged_angles.csv'
# csv_file = getDataFile()
data_dict = loadCsv(csv_file, 'packet', 'time_us', 'angle')
data = classify('packet', data_dict)

# Make the problem "deterministic" (same random seed for each run)
np.random.seed(1993)

# Do we want static friction? If not, it will be assumed to be zero
with_static_friction = False

# frequency and order of the butterworth filter used in smoothing
fc = 15
Nf = 4

# The problem has to be written as A*X=B, the following does precisely this!
RegA = np.zeros((0, 3))
RegB = np.zeros((0, 1))

for test_id, test_data in data.items():
    t = test_data.get('time_us')
    t = t / 1e6
    t = t - t[0]  # t[0]=0
    tv = t[1:-1]  # for velocity and acceleration signals
    dt = np.average(np.diff(t))
    # raw signals
    praw = test_data.get('angle') * (2 * np.pi / 4000)
    vraw = np.convolve(praw, [0.5, 0, -0.5], 'valid') / dt
    araw = np.convolve(praw, [1.0, -2.0, 1.0], 'valid') / (dt ** 2)
    # filter the signals
    b, a = signal.butter(Nf, 2 * (dt * fc))  # [1],[1,0] = no filter
    p = signal.filtfilt(b, a, praw, padtype=None)
    v = signal.filtfilt(b, a, vraw, padtype=None)
    a = signal.filtfilt(b, a, araw, padtype=None)
    # cut left/right ends (to avoid issues with the butterworth filter)
    ncut = 2 * Nf + 1
    P = p[ncut + 1:-(ncut + 1)]
    V = v[ncut:-ncut]
    A = a[ncut:-ncut]
    # select just some samples to avoid overfitting
    keep_ratio = 5
    keep_size = round(P.size / keep_ratio)
    keep_this = np.random.choice(P.size, size=(keep_size,), replace=False)
    P = P[keep_this].reshape(-1, 1)
    V = V[keep_this].reshape(-1, 1)
    A = A[keep_this].reshape(-1, 1)
    # Extend the regression matrix and vector
    RegA = np.vstack((RegA, np.hstack((g * np.sin(P), V, np.sign(V)))))
    RegB = np.vstack((RegB, -A))

# solve the problem  min || RegY*X - RegB ||
if with_static_friction:
    X = np.linalg.lstsq(RegA, RegB, rcond=None)[0].reshape(-1)
else:
    Xred = np.linalg.lstsq(RegA[:, :2], RegB, rcond=None)[0].reshape(-1)
    X = np.append(Xred, 0)

# Extract "real parameters":
I0 = ((m2 / 2 + m3) * ell) / X[0]
I = I0 - (m2 / 4 + m3) * ell ** 2
tau_d = X[1] * I0
tau_s = X[2] * I0
print("I:", I)
print("tau_d:", tau_d)
print("tau_s:", tau_s)