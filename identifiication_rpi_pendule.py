'''
start script make data acquisitions by
'''
#!/usr/bin/env python
import time
from rotary_encoder import decoder,decoder_quadrature
from matplotlib import pyplot as plt
import logging
from logging import info, basicConfig
import pigpio
from pynput import keyboard
import sys
import numpy as np
from threading import Lock
lock = Lock()


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

counter=0
STOP = True
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)	
	global STOP, counter
	lock.acquire()
	if not STOP:
		counter+=1
	print(counter)
	STOP = not STOP
	lock.release()
	
ENCODER_GLITCH=0	
pi.set_glitch_filter(17, 80)
pi.set_glitch_filter(18, 80)
pi.set_glitch_filter(19, ENCODER_GLITCH)
pi.set_glitch_filter(26, ENCODER_GLITCH)
pi.set_glitch_filter(20, ENCODER_GLITCH)
pi.set_glitch_filter(21, ENCODER_GLITCH)
 

start_time = time.time()
posPendule = 180
oldPosPendule=posPendule
arrPosPendule = [posPendule]
times=[time.time()-start_time]
timesVitesseP=[]
arrVitesseP=[0.0]
arrAccPendule=[0.0]
Te=0.01
def callbackPendule(way):
	global posPendule,start_time
	posPendule -= way/1200*180
	
def acquisitions(Te=0.05):
	if time.time()-start_time-times[-1]>Te: #Te temps ecoule	
		arrPosPendule.append(posPendule)
		getVitPendule()
		times.append(time.time()-start_time)
		#info("pos pendule={}".format(posPendule))#*180/600
			

arrAccPendule=[]	
def getVitPendule():
	global oldPosPendule, posPendule, Te 
	vitPendule = (posPendule-oldPosPendule)/(Te)
	oldPosPendule = posPendule
	 
if __name__ == "__main__":
	try:
		cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
		cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
		decoderPendule = decoder_quadrature(pi, 19, 26, callbackPendule) #pendule
		
		oldStop=0
		old_time_pendule=time.time()
		times=[]
		posPenduleArr=[]
		counters=[]
		start_time=time.time()		
		while 1:
			lock.acquire()			
			if STOP:
				lock.release()				
				continue
			lock.release()
			print('data')
			counters.append(counter)
			posPenduleArr.append(posPendule)
			times.append(time.time()-start_time)
			time.sleep(Te)
			
			
			
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  0)
		 
		'''
		print(arrPosPendule)		
		plt.plot(times,arrPosPendule,'b.')
		plt.plot(times,arrVitesseP,'r.')
		#plt.plot(times,arrAccPendule,'g-')
		plt.xlabel('time in s')
		plt.ylabel('en degree')
		plt.show()'''
	finally:
		np.savetxt('angle_iden.csv',[counters,times,posPenduleArr],delimiter=',')
		info('exc')
		
plt.plot(times,posPenduleArr,'b.')
plt.xlabel('time in s')
plt.ylabel('en degree')
plt.savefig('angle_iden.png',dpi=200)

decoderPendule.cancel()
cb1.cancel()
cb2.cancel()
print('total program time: ',start_time-time.time())
time.sleep(20000)
