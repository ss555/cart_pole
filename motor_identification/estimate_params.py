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
    for filename in glob.glob("./motor_identification/data-old/pwm-*.csv"):
        pwm = int(re.search('(\\d+)', filename).group(1))
        result.append((pwm, filename))
    return result

def read_params(pwm):
    """
        [i, pwm, time, x, v]
    """
    # if pwm > 0:
    if pwm < 0:
        filename = "./motor_identification/data/pwm%d.csv" % pwm
    else:
        filename = "./motor_identification/data/pwm-rev%d.csv" % (-pwm)
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(list(map_to_float(row)))
    data = np.array(data)
    print(data.shape)
    rows, cols = data.shape
    ext = np.zeros((rows, cols + 1))
    ext[:,:-1] = data    
    for i in range(1, rows):
        dt = (ext[i, 2] - ext[i - 1, 2]) / 1.0
        ext[i, -1] = (ext[i, 3] - ext[i - 1, 3]) / dt
    return ext

def read_all():
    result = {}
    pwms = [-200, -175, -150, -125, -100, -75, -50, 50, 75, 100, 125, 150, 175, 200]
    # pwms = [100]
    for pwm in pwms: #get_filenames_and_pwm():
        data = read_params(pwm)
        result[pwm] = data
    return result

def plot_velocity(data):
    for value in sorted(data.values(), key=lambda x: -x[0, 1]):
        times = value[:,2] / 1.0        
        u = - 12.0 * value[0, 1] / 255
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
    # print(data-old)
    for a in np.arange(13.0, 15.0, 0.02):
        print("Fitting for a = %.2f; error = %.2f" % (a, _error))
        for b in np.arange(-1.0, 0.0, 0.02):
            for c in np.arange(-1.0, 0.0, 0.02):
                error = 0.0
                for item in data.values():
                    pwm = item[0, 1]
                    u = 12.0 * pwm / 255 # inverse pwm  -
                    times = item[:, 2] / 1.0
                    curve = integrate_curve(a, b, c, u, times)
                    #dv = -a * vs[i] + b * u + c * np.sign(vs[i])
                    error = error + get_square_error(item[:, 4], curve)
                if error < _error:
                    _error = error
                    _a, _b, _c = a, b, c
                    #print("Val %.2f pour a=%.2f b=%.2f et c=%.2f"% (error,a,b,c)
                    print("a=%.2f, b=%.2f, c=%.2f, new err = %.3f et best error=%.3f" % (a,b,c,error,_error))
    return _a, _b, _c


all_data = read_all()
# f_a,f_b,f_c = fit_params(all_data)
f_a=13.74
f_b=-0.72
f_c=-0.58

# a=13.74, b=-0.72, c=-0.58
# dv = -a * vs[i] + b * u + c * np.sign(vs[i]), donc u=(dv+avs[i]-c*np.sign(vs[i]))/b
# a=13.90, b=-0.74, c=-0.70, new err = 0.06
# force statique: fc/fb=0.68/0.72/12*255=20.1 en PWM
# force statique: fc/fb=0.58/0.72/12*255=17.1 en PWM
# dv = -a * vs[i] + b * u + c * np.sign(vs[i]), F=ma=m*dv/dt,
# #Tension = (fcontrol + fA * fvit_chariot + copysignf (fC, fvit_chariot)) / fB;
# max_force : 180/255*12=8.47, force_max=-a * vs[i] + b * u - c * np.sign(vs[i])=-0.72*8.47+0.58=-5.52
#f dynamique f_a/f_b=19
#print fit_params(all_data)

plot_velocity(all_data)
plot_set_velocity(all_data)



