from time import time
pendulePy.sendCommand(50)
s=
for i in range(150):
    pendulePy.readState(blocking=True)
    print(pendulePy.linvel)