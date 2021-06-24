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

lengthLimit=5500
posChariot = 0
arrTensions=[]
timesTension=[]
times=[]
posArrChariot=[]


posPendule = 300
arrPosPendule = []
timePosPendule = []
oldPosChariot=0
timesVitesse=[]
arrVitesse=[]
Te=0.010
def getVitChariot():
	global posChariot, oldPosChariot, old_time_chariot
	if time.time()-old_time_chariot>Te:
		arrVitesse.append((posChariot-oldPosChariot)/(time.time()-old_time_chariot))
		timesVitesse.append(time.time()-start_time)    
		old_time_chariot = time.time()
		oldPosChariot = posChariot
    
def callbackPendule(way):
	global posPendule
	posPendule += way
	arrPosPendule.append(posPendule)
	timePosPendule.append(time.time()-start_time)
	info("pos pendule={}".format((posPendule%600)*0.3))#*180/600
def getVitPendule():
	global posPendule, oldPosPendule, old_time_pendule
	vitPendule = ((posPendule-oldPosPendule)/(time.time()-old_time_pendule))
	old_time_pendule=time.time()
	oldPosPendule = posPendule
	return vitPendule
ENCODER_GLITCH=50	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)
pi.set_glitch_filter(19, ENCODER_GLITCH)
pi.set_glitch_filter(26, ENCODER_GLITCH)
pi.set_glitch_filter(20, 5)
pi.set_glitch_filter(21, 5)
def callbackChariot(way):
	global posChariot
	global lengthLimit
	posChariot += way
	info("pos chariot={}".format(posChariot))
	posArrChariot.append(posChariot)
	times.append(time.time()-start_time)
	getVitChariot() 
if __name__ == "__main__":
	cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
	cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
	decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
	decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
	try:
		start_time = time.time()
		#homeCall=pi.event_callback(1,homing)
		# sens encodeur +6250:left 0-middle -6250: right
		info('power')
		print('start')
		old_time_chariot=time.time()		
		#HYPOTHESE!!!: start in middle
		info('posChariot initiale: {}'.format(posChariot))
		PWM=70
		sleep_time=0.4
		for i in range(3):
			pi.write(16,1); #1-right 0-left
			pi.set_PWM_dutycycle(24,  PWM)
			arrTensions.append(-75)
			timesTension.append(time.time()-start_time) 
			time.sleep(sleep_time)
			arrTensions.append(-75)
			timesTension.append(time.time()-start_time)
			pi.write(16,0); #1-right 0-left
			pi.set_PWM_dutycycle(24,  PWM)
			arrTensions.append(75)
			timesTension.append(time.time()-start_time) 
			time.sleep(sleep_time)
			arrTensions.append(75)
			timesTension.append(time.time()-start_time) 			
		arrTensions.append(75/255)
		timesTension.append(time.time()-start_time)
		
		
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
plt.plot(times,posArrChariot,'b',label='position chariot')
plt.plot(timesTension,arrTensions,label='tension chariot')
plt.plot(timesVitesse,arrVitesse,'r.',label='vitesse chariot')
plt.legend(('position chariot','tension chariot','vitesse chariot'),loc='upper right')
plt.xlabel('time in s')
plt.ylabel('pos')
plt.title('PWM:'+str(PWM))
plt.show() 
pi.stop()
	
fig, ax = plt.subplots()
ax.plot(timePosPendule, arrPosPendule)
ax.set(xlabel='time (s)', ylabel='angle (degrees)', title='Pendulum angle evolution with time')
ax.grid()

#fig.savefig("test.png")
plt.show()

print('total program time: ',start_time-time.time())
