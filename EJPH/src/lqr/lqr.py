'''
help info: underactuated mit continuous_vs_discrete_time.ipynb
'''
#!/usr/bin/env python3
from src.pendule_pi import PendulePyClient
import math
import time
import sys

def normalize(angle):
  while angle > math.pi:
    angle -= 2*math.pi
  while angle < -math.pi:
    angle += 2*math.pi
  return angle


def sign(x):
  return 1 if x >= 0 else -1


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage:", sys.argv[0], "host-name")
    sys.exit(0)
  pendulum = PendulePyClient(wait=5, host=sys.argv[1])

  MAX_ANGLE = 0.1
  kp = -127.45880662905581
  kpd = -822.638944546691
  kt = 2234.654627319883
  ktd = 437.1177135919267
  
  kswing = 70.0
  kx = 0.0

  try:
    while True:
      pendulum.readState(blocking=True)
      et = normalize(pendulum.angle-math.pi)
      
      if abs(et) < MAX_ANGLE:
        pwm = - kp * pendulum.position - kt * et - kpd * pendulum.linvel - ktd * pendulum.angvel
      else:
        ks = kswing * (0.1 + 0.9 * 0.5 * (1 - math.cos(et)))
        pwm = ks * (1-math.cos(et)) * sign(pendulum.angvel * math.cos(et)) - kx * pendulum.position
      pendulum.sendCommand(pwm)
  except KeyboardInterrupt:
    pendulum.sendCommand(0)
    time.sleep(1)
