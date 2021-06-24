import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as pp

from math import pi

from dc_motor_equations import integrate_curve

def map_to_float(a):
    return map(lambda x: float(x), a)

def get_filenames_and_pwm():
    result = []
    for filename in glob.glob("./data/pwm-*.csv"):
        pwm = int(re.search('(\\d+)', filename).group(1))
        result.append((pwm, filename))
    return result

def read_params(pwm):
    """
        [i, pwm, time, x, v]
    """
    if pwm > 0:
        filename = "./data/pwm-%d.csv" % pwm
    else:
        filename = "./data/pwm-rev-%d.csv" % (-pwm)
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(map_to_float(row))
    data = np.array(data)
    rows, cols = data.shape
    ext = np.zeros((rows, cols + 1))
    ext[:,:-1] = data    
    for i in range(1, rows):
        dt = (ext[i, 2] - ext[i - 1, 2]) / 1.0
        ext[i, -1] = (ext[i, 3] - ext[i - 1, 3]) / dt
    return ext

def read_all():
    result = {}
    pwms = [-250, -225, -200, -150, -100, -75, -50, 50, 75, 100, 150, 200, 225, 250]
    # pwms = [100]
    for pwm in pwms: #get_filenames_and_pwm():
        data = read_params(pwm)
        result[pwm] = data
    return result

def plot_velocity(data):
    for value in sorted(data.values(), key=lambda x: -x[0, 1]):
        times = value[:,2] / 1.0        
        u = 12.0 * value[0, 1] / 255
        curve = integrate_curve(f_a, f_b, f_c, u, times)
        pp.plot(times[:100], value[:, 4][:100], label=("%.1fV" % u))
        pp.plot(times[:100], curve[:100])
    pp.legend()
    pp.xlabel("Time, s")
    pp.ylabel("Velocity, m/s")
    pp.grid(True)
    pp.show()

def plot_set_velocity(data):
    set_vs = np.zeros((len(data), 2))
    for i, value in enumerate(data.values()):
        velocities = value[-10:-1, 4]
        set_vs[i, 0] = value[0, 1]
        set_vs[i, 1] = sum(velocities) / len(velocities)

    pp.plot(set_vs[:, 0], set_vs[:, 1], 'o')
    pp.grid(True)
    pp.show()


def get_square_error(a1, a2):
    return sum(map(lambda x: x * x, a1 - a2))


def fit_params(data):
    _a = 0.0
    _b = 0.0
    _c = 0.0
    _error = float('inf')

    # print(data)

    for a in np.arange(13.0, 15.0, 0.02):
        print("Fitting for a = %.2f; error = %.2f" % (a, _error))
        for b in np.arange(0., 1.5, 0.02):
            for c in np.arange(-1.0, 1.0, 0.02):
                error = 0.0
                for item in data.values():
                    pwm = item[0, 1]
                    u = 12.0 * pwm / 255
                    times = item[:, 2] / 1.0
                    curve = integrate_curve(a, b, c, u, times)
                    error = error + get_square_error(item[:, 4], curve)
                if error < _error:
                    _error = error
                    _a, _b, _c = a, b, c
                    #print("Val %.2f pour a=%.2f b=%.2f et c=%.2f"% (error,a,b,c)
                    print("a=%.2f, b=%.2f, c=%.2f, new err = %.2f et best error=%.2f" % (a,b,c,error,_error))
    return _a, _b, _c


all_data = read_all()

# (19.0, 0.95000000000000007, -0.29999999999999849) New moi

#(21.45999999999999, 1.1400000000000001, -0.81999999999999895)  Moi
#35.98, 2.22, -2.79) Russe

f_a = 14.3#14.05 #21.46#16.2
f_b = 0.76#0.75#1.14#0.05
f_c = -0.44#-0.82#-2.9

#print fit_params(all_data)

plot_velocity(all_data)
plot_set_velocity(all_data)

# Fit params:
# 1.4, 0.09, 0.04

# pwm = 100
# u = 12.0 * pwm / 255
# data = all_data[pwm]
# times = data[:, 2] / 1000000.0
# curve = integrate_curve(f_a, f_b, f_c, u, times)
# pp.plot(times, data[:, 4])
# pp.plot(times, curve)
# pp.show()



