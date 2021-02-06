#!/usr/bin/env python
import time
from matplotlib import pyplot as plt
from rotary_encoder import decoder
import logging
from logging import info, basicConfig
import pigpio
from pynput import keyboard
import sys
PI_INPUT = pigpio.INPUT
PI_PUD_UP = pigpio.PUD_UP
pi = pigpio.pi()
if not pi.connected:
    exit()
    print('exit')
    
    
logname='DEBUG'
basicConfig(filename=logname,
			filemode='w',#'a' for append
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)
    
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
	#raise Exception('capteur fin course') 	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)



posPendule = 300
arrPosPendule = []
timePosPendule = []
def callbackPendule(way):
	global posPendule
	posPendule += way
	arrPosPendule.append(posPendule)
	timePosPendule.append(time.time()-start_time)
	info("pos pendule={}".format((posPendule%600)*0.3))#*180/600
  
lengthLimit=5500
posChariot = 0
def callbackChariot(way):
	global posChariot
	global lengthLimit
	posChariot += way
	info("pos chariot={}".format(posChariot))
	if posChariot > lengthLimit or posChariot < -lengthLimit:
		pass
		#declare an event 1
		#pi.event_trigger(1)
		#info('event#1 happened') 
def homing(event, tick): # use for event 1
	print('going center')
	global posChariot
	global lengthLimit
	if posChariot > lengthLimit:		
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  50) #max 255
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
		start_time = time.time()		
		#HYPOTHESE!!!: start in middle
		info('posChariot initiale: {}'.format(posChariot))
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  50) 
		while 1:
			#main			
			#if posChariot>-5500: # right
			#if posChariot<5500: # left
			time.sleep(0.01)
			
			if posChariot<=-6300:
				print('reached')
				break
				
			PWM_magnitude = 180			
			
		
		
		#time.sleep(20)
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  0)
		 
		
	except:
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
'''
pi.write(16,1);
		pi.set_PWM_dutycycle(24,  100)
		time.sleep(50)
		
 total length 12500(90 cm) ticks Chariot, it takes 8.8 sec with PWM=50 to go full length
 
 * moteur (signal PWM) : fil jaune => GPIO24 gpioPWM(24,0);
 * moteur (signal DIR) : fil vert => GPIO16 gpioWrite(16,0);
 
 direction looking from raspberry
 right: pi.write(16,1);
 left: pi.write(16,0);
 
 sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
 stable-baselines
 
 tensorflow==1.14
 gym
 
pi.write(16,0); #1-right 0-left
pi.set_PWM_dutycycle(24,  40) #max 255		
time.sleep(20)
pi.write(16,1); #1-right 0-left
pi.set_PWM_dutycycle(24,  10) #max 255		
time.sleep(1)
'''

'''
def goCenter():
	global posChariot
	if position>5500:
		pi.write(16,0);
		pi.set_PWM_dutycycle(24,  50) #max 255
		while position>0:
			continue
	else:
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  50) #max 255
		while position<0:
			continue
	pi.set_PWM_dutycycle(24,  0) #max 255
	
	
	
if posChariot>6250:
		pi.set_PWM_dutycycle(24,  0)
		break
		
while 1:
	if abs(posChariot)>5500:
		goCenter(posChariot)
		break
'''
