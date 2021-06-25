#!/usr/bin/env python3
#wifi latency aller-retour can go up to 0.1s, alors que par fil: moins de 1ms
import socket
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifi-hotspot  # The server's hostname or IP address
PORT = 65432        # The port used by the server
import math
import numpy as np
#rpi
import time
from matplotlib import pyplot as plt
from rotary_encoder import decoder
from rotary_encoder_adapted import decoderFour
import logging
from logging import info, basicConfig
import pigpio
import sys
from threading import Lock
FORMAT='utf-8'
PI_INPUT = pigpio.INPUT
PI_PUD_UP = pigpio.PUD_UP
pi = pigpio.pi()
lock = Lock()
if not pi.connected:
    exit()
    print('exit')
    
logname='TCP_DEBUG'
basicConfig(filename=logname,
			filemode='w',#'a' for append
			format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
			datefmt='%H:%M:%S',
			level=logging.DEBUG)			
start_time = time.time()
#chariot
posChariot = 0
oldPosChariot = 0
#pendule
posPendule = math.pi
oldPosPendule = math.pi

#time
old_time_chariot=time.time()
old_time_pendule=time.time()
reset_time_start=0
done=0
reset_counter=0
#constants
Te=0.05
RESET_TIME=90
TIME_LIMIT=30000#TIME_LIMIT=2000
rPWM=60
SCALE=180#Puissance
COMPENSATE_STATIC=0
ENCODER_GLITCH=0
def cbf(gpio, level, tick):
	lock.acquire()	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()
	info('fin course')
	lock.release()	
pi.set_glitch_filter(17, 0)
pi.set_glitch_filter(18, 0)
pi.set_glitch_filter(19, ENCODER_GLITCH)
pi.set_glitch_filter(26, ENCODER_GLITCH)
pi.set_glitch_filter(20, ENCODER_GLITCH)
pi.set_glitch_filter(21, ENCODER_GLITCH)

def callbackPendule(way):
	global posPendule
	posPendule+=way/300*math.pi     
       
def getVitChariot():
	global posChariot, oldPosChariot    
	vitChariot = (posChariot-oldPosChariot)/Te
	oldPosChariot = posChariot
	return vitChariot
    
def getVitPendule():
	global posPendule, oldPosPendule, Te
	vitPendule = (posPendule-oldPosPendule)/Te
	oldPosPendule = posPendule
	return vitPendule
            
def callbackChariot(way):
	global posChariot, done	 
	posChariot += way*0.00006671695598673524  #*2*PI/600*0.006371
	# sens encodeur +6250:left 0-middle -6250: right	
	if posChariot>0.37 and not done:		
		done=1
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM)
	elif posChariot<-0.37 and not done:
		done=1
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM)
		
        
def applyForce(action):
	# sens encodeur +6250:left 0-middle -6250: right
	global SCALE
	if action > 0 and action <=1.0:
		pi.write(16,0); #1-right 0-left
	elif action>=-1.0:
		pi.write(16,1); #1-right 0-left
	else:		
		print(f'error command {action}')
		pi.set_PWM_dutycycle(24, 0)
		raise Exception
	lock.acquire()
	pi.set_PWM_dutycycle(24, SCALE*action+COMPENSATE_STATIC)
	lock.release()
    
def getState():
	global posChariot, posPendule, done 
	vChariot=getVitChariot()
	vPendule=getVitPendule()
	#clippedPosPendule=math.atan2(math.sin(posPendule),math.cos(posPendule))
	info('got state: {}'.format([posChariot, vChariot, math.cos(posPendule), math.sin(posPendule), vPendule, done]))    
	return [posChariot, vChariot, math.cos(posPendule), math.sin(posPendule), vPendule, done]
    
def initVariables():
	global posChariot,oldPosChariot,posPendule,oldPosPendule,old_time_chariot,old_time_pendule,done,send_time
	#pendule
	posPendule = math.pi
	oldPosPendule = math.pi
	old_time_chariot=time.time()
	old_time_pendule=time.time()
	done=0
	send_time=time.time()

def reset():
	global posChariot,oldPosChariot,posPendule,oldPosPendule,old_time_chariot,old_time_pendule,done, reset_time_start ,RESET_TIME, reset_counter
	# GOTO center slowly
	done=0
	info('reset')
	data_state = getState()
	x=data_state[0]
	reset_time_start=time.time()	
	if x > 0:
		pi.write(16,1); #1-right 0-left
		lock.acquire()
		pi.set_PWM_dutycycle(24,  rPWM) #max 255
		lock.release()
		while x>0:                
			time.sleep(0.00001)
			x,_, _,_,_,_ = getState()	
	elif x < 0:
		pi.write(16,0); #1-right 0-left
		lock.acquire()
		pi.set_PWM_dutycycle(24,  rPWM) #max 255
		lock.release()		
		while x<0:
			time.sleep(0.00001)
			x,_, _,_,_,_ = getState()
	pi.set_PWM_dutycycle(24,  0)
	reset_counter+=1
	print(reset_counter)
	if reset_counter%10==0 and reset_counter>1:
		print('restarting in init values')	
		while time.time()-reset_time_start<RESET_TIME:
			time.sleep(0.01)
		initVariables()
	print('initialised variables')
def sendState():
	data_state = getState()				
	data_send = ','.join(str(data_state[i]) for i in range(len(data_state)))				
	s.sendall(data_send.encode(FORMAT))
if __name__ == '__main__':
	try:
		
		cb1 = pi.callback(17, pigpio.FALLING_EDGE, cbf)
		cb2 = pi.callback(18, pigpio.FALLING_EDGE, cbf)
		decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
		decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			info('connecting')
			s.connect((HOST, PORT))
			initVariables()
			send_time=time.time()			  
			while True:								
				data_action = s.recv(32).decode(FORMAT)
				info('action: {}'.format(data_action))				
				if str(data_action)=='RESET':
					reset()					
				else:
					try:
						applyForce(float(data_action))
					except:
						print('force {}'.format(data_action))					
					
				if time.time()-send_time<Te:
					time.sleep(Te-time.time()+send_time)#Te		
				sendState()
				send_time=time.time()
	finally:        
		pi.set_PWM_dutycycle(24,  0)
		cb1.cancel()
		cb2.cancel()		
		decoderPendule.cancel()
		decoderChariot.cancel()
		pi.stop()
		logging.exception("Fatal error in main loop")
		getState()
		info('smth went wrong')
		info("time passed: {}".format(time.time()-start_time))
		s.close()
 