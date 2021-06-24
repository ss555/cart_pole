'''
THIS PROGRAM TO DUMP DATA to csv file.

'''
#!/usr/bin/env python
import numpy as np
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
start_time = time.time()    
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

def cbf(gpio, level, tick):
	lock.acquire()	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()
	lock.release()	
pi.set_glitch_filter(17, 20)
pi.set_glitch_filter(18, 20)
pi.set_glitch_filter(20, 0)
pi.set_glitch_filter(21, 0)


def applyForce(PWM):
	lock.acquire()
	pi.set_PWM_dutycycle(24,PWM)
	lock.release()
posChariot = 0
def callbackChariot(way):
	global posChariot
	posChariot += way*0.00006671695598673524
	info("pos chariot={}".format(posChariot))
if __name__ == "__main__":
	cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
	cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
	decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
	tension=[]
	position=[]
	times=[]
	flag=0
	try:
		
		# sens encodeur +6250:left 0-middle -6250: right
		info('power')
		print('start')
				
		info('posChariot initiale: {}'.format(posChariot))
		#PWM initiale
		PWM=180
		pi.write(16,0); #1-right 0-left

		start_time=time.time()
		for i in range(3):
			applyForce(PWM)
			while posChariot<0.5:
				position.append(posChariot)
				times.append(time.time()-start_time)
				time.sleep(0.05)
			tension=np.append(tension,np.ones(len(times)-flag)*(PWM))
			flag=len(times)
			pi.write(16,1);	
			while posChariot>0:
				position.append(posChariot)
				times.append(time.time()-start_time)
				time.sleep(0.05)
			tension=np.append(tension,np.ones(len(times)-flag)*(-PWM))
			flag=len(times)
			pi.write(16,0);
			#PWM=PWM+10
							
		#time.sleep(20)
		pi.set_PWM_dutycycle(24,  0);
		#print
		np.savetxt("chariot_iden.csv",[tension,times,position],delimiter=",")
		
	finally:
		info('smth went wrong')
		# en cas de CTRL+C ou ERREUR dans le programme mise a zero Pmoteur
		pi.set_PWM_dutycycle(24,  0)
		cb1.cancel()
		cb2.cancel()		
		decoderChariot.cancel()
		pi.stop()

print('total program time: ',start_time-time.time())

