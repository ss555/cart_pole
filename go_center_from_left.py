#!/usr/bin/env python
import time
from matplotlib import pyplot as plt
from rotary_encoder import decoder
import logging
from logging import info, basicConfig
import pigpio
from pynput import keyboard
import sys
from threading import Lock
PI_INPUT = pigpio.INPUT
PI_PUD_UP = pigpio.PUD_UP
pi = pigpio.pi()
if not pi.connected:
    exit()
    print('exit')
lock=Lock()    
    
logname='DEBUG'
basicConfig(filename=logname,
			filemode='w',#'a' for append
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)
start_time = time.time()    
info("STARTING")   
#Fin de course
pi.set_mode(17, PI_INPUT); #//FDC17 
pi.set_mode(18, PI_INPUT); #//FDC18
pi.set_pull_up_down(17, PI_PUD_UP);   
pi.set_pull_up_down(18, PI_PUD_UP); 

oldIFinCourseMoteur = pi.read(18);	#//1 si pas appui
oldIFinCourseEncodeur = pi.read(17);  #//1 si pas appui
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()	
pi.set_glitch_filter(17, 40)
pi.set_glitch_filter(18, 40)



posPendule = 300
arrPosPendule = []
timePosPendule = []
def callbackPendule(way):
	global posPendule
	posPendule += way
	arrPosPendule.append(posPendule)
	timePosPendule.append(time.time()-start_time)
  
lengthLimit=5500
posChariot = 0
def callbackChariot(way):
	global posChariot
	global lengthLimit
	posChariot += way
	info("pos chariot={}".format(posChariot))
	if posChariot > lengthLimit or posChariot < -lengthLimit:
		pass
def homing(event, tick): # use for event 1
	print('going center')
	global posChariot
	global lengthLimit
	if posChariot > lengthLimit:		
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  38) #max 255
		while 1:		
			if posChariot<0:
				pi.set_PWM_dutycycle(24,  0)
				break
	elif posChariot < -lengthLimit:
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  50) #max 255
		while 1:		
			if posChariot>0:
				pi.set_PWM_dutycycle(24,  0)
				break	
if __name__ == "__main__":
	cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
	cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
	decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
	decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
	try:
		
		#homeCall=pi.event_callback(1,homing)
		# sens encodeur +6250:left 0-middle -6250: right
		info('power')
		print('start')
				
		#HYPOTHESE!!!: start in middle
		info('posChariot initiale: {}'.format(posChariot))
		pi.write(16,1); #1-right 0-left
		lock.acquire()
		pi.set_PWM_dutycycle(24,  60)
		lock.release() 
		while 1:
			time.sleep(0.01)
			if posChariot<=-6300*9:
				print('reached')
				break
			PWM_magnitude = 180			
		#time.sleep(20)
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  0)		
	finally:
		info('smth went wrong')
		# en cas de CTRL+C ou ERREUR dans le programme mise a zero Pmoteur
		pi.set_PWM_dutycycle(24,  0)
cb1.cancel()
cb2.cancel()		
decoderPendule.cancel()
decoderChariot.cancel()
homeCall.cancel() 
pi.stop()	
fig, ax = plt.subplots()
ax.plot(timePosPendule, arrPosPendule)
ax.set(xlabel='time (s)', ylabel='angle (degrees)', title='Pendulum angle evolution with time')
ax.grid()

#fig.savefig("test.png")
plt.show()

print('total program time: ',start_time-time.time())
