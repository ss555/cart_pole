#!/usr/bin/env python
import time
from rotary_encoder import decoder
from matplotlib import pyplot as plt
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
STOP = 0
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)	
	global STOP
	STOP = 1
	sys.exit(0)
	
	
ENCODER_GLITCH=0	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)
pi.set_glitch_filter(19, ENCODER_GLITCH)
pi.set_glitch_filter(26, ENCODER_GLITCH)
pi.set_glitch_filter(20, 0)
pi.set_glitch_filter(21, 0)

cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)  

start_time = time.time()
posPendule = 180
oldPosPendule=posPendule
arrPosPendule = [posPendule]
times=[time.time()-start_time]
timesVitesseP=[]
arrVitesseP=[0.0]
arrAccPendule=[0.0]
def callbackPendule(way):
	global posPendule,start_time
	posPendule += way/300*180
	
def acquisitions(Te=0.05):
	if time.time()-start_time-times[-1]>Te: #Te temps ecoule	
		arrPosPendule.append(posPendule)
		getVitPendule()
		times.append(time.time()-start_time)
		#info("pos pendule={}".format(posPendule))#*180/600
			

arrAccPendule=[]	
def getVitPendule():
	global oldPosPendule,posPendule,old_time_pendule,arrVitesseP,arrAccPendule
	vitPendule = ((posPendule-oldPosPendule)/(0.05))	
	arrAccPendule.append((vitPendule-arrVitesseP[-1])/(0.05))
	#old_time_pendule=time.time()
	arrVitesseP.append(vitPendule)
	oldPosPendule = posPendule	
	#info("pos pendule={}".format(vitPendule))
	#return vitPendule
	 
posChariot = 0
def callbackChariot(way):
	global posChariot
	posChariot += way
	info("pos chariot={}".format(posChariot))
	

	
def oscillations(amplitude,n_repetitions=2,PWM_magnitude=50):
	global posChariot	
	for i in range(n_repetitions):
		info('going left')
		#GO LEFT		
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  PWM_magnitude)		
		while posChariot<amplitude:
			time.sleep(0.001)		
		info('going right')	
		#GO RIGHT
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  PWM_magnitude)
		while posChariot>-abs(amplitude):
			time.sleep(0.001)
if __name__ == "__main__":
	try:
		cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
		cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
		decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
		#decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
		# sens encodeur +6250:left 0-middle -6250: right
		info('power')
		old_time_pendule=time.time()		
		while 1:
			acquisitions()
			#print(posPendule)			
			if STOP:
				info('arret d\'urgence')
				pi.set_PWM_dutycycle(24,  0)
				print('break')
				break
		
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  0)
		 
		
		print(arrPosPendule)		
		plt.plot(times,arrPosPendule,'b.')
		plt.plot(times,arrVitesseP,'r.')
		#plt.plot(times,arrAccPendule,'g-')
		plt.xlabel('time in s')
		plt.ylabel('en degree')
		plt.show()
	finally:
		
		info('smth went wrong')
		# en cas de CTRL+C ou ERREUR dans le programme mise a zero Pmoteur
		pi.set_PWM_dutycycle(24,  0)
		decoderPendule.cancel()
		cb1.cancel()
		cb2.cancel()



print('total program time: ',start_time-time.time())
time.sleep(20000)
