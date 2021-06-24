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
import logging
from logging import info, basicConfig
import pigpio
import sys
FORMAT='utf-8'
PI_INPUT = pigpio.INPUT
PI_PUD_UP = pigpio.PUD_UP
pi = pigpio.pi()

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
#temp vars
counter=0
reset_time_start=0
done=0
#constants
RESET_TIME=60
TIME_LIMIT=30000#TIME_LIMIT=2000
rPWM=100
SCALE=40#Puissance
COMPENSATE_STATIC=40
ENCODER_GLITCH=200
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)
	pi.stop()
	info('fin course')
	raise Exception('capteur fin course')	
pi.set_glitch_filter(17, 50)
pi.set_glitch_filter(18, 50)
pi.set_glitch_filter(19, ENCODER_GLITCH)
pi.set_glitch_filter(26, ENCODER_GLITCH)
pi.set_glitch_filter(20, 10)
pi.set_glitch_filter(21, 10)

def callbackPendule(way):
    global posPendule
    posPendule+=way/300*math.pi
    posPendule=math.atan2(math.sin(posPendule),math.cos(posPendule))
    #info("pos pendule={}".format(posPendule))#*180/600  
    

    
def getVitChariot():
    global posChariot, oldPosChariot, old_time_chariot    
    vitChariot = ((posChariot-oldPosChariot)/(time.time()-old_time_chariot))
    old_time_chariot = time.time()
    oldPosChariot = posChariot
    return vitChariot
    
def getVitPendule():
	global posPendule, oldPosPendule, old_time_pendule
	if (posPendule-oldPosPendule)>5:#because of clipping posChariot values to 0;2pi
		vitPendule = ((posPendule-(oldPosPendule+math.pi*2))/(time.time()-old_time_pendule))
		info('1 tour')
	elif (oldPosPendule-posPendule)>5:		
		info('1 tour')
		vitPendule = ((posPendule+math.pi*2-oldPosPendule)/(time.time()-old_time_pendule))
	else:
		vitPendule = ((posPendule-oldPosPendule)/(time.time()-old_time_pendule))
	old_time_pendule=time.time()
	oldPosPendule = posPendule
	return vitPendule
            
def callbackChariot(way):
	global posChariot, done	 
	posChariot += way/1000
	
	# sens encodeur +6250:left 0-middle -6250: right	
	if posChariot>5 and not done:		
		done=1
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM)
	elif posChariot<-5 and not done:
		done=1
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM)
		
        
def applyForce(action):
	global SCALE
	if action > 0:
		pi.write(16,1); #1-right 0-left
	else:
		pi.write(16,0); #1-right 0-left
	pi.set_PWM_dutycycle(24, abs(action)*SCALE + COMPENSATE_STATIC)
    
    
def getState():
    global posChariot, posPendule, done , counter, TIME_LIMIT
    vChariot=getVitChariot()
    vPendule=getVitPendule()
    info('got state: {}'.format([posChariot, vChariot, posPendule, vPendule]))
    
    return [posChariot, vChariot, posPendule, vPendule, counter/TIME_LIMIT, done]
def initVariables():
	global posChariot,oldPosChariot,posPendule,oldPosPendule,old_time_chariot,old_time_pendule,counter,done
	#pendule
	posPendule = math.pi
	oldPosPendule = math.pi
	counter=0
	old_time_chariot=time.time()
	old_time_pendule=time.time()
	done=0
def reset():
	global posChariot,oldPosChariot,posPendule,oldPosPendule,old_time_chariot,old_time_pendule,counter,done, reset_time_start ,RESET_TIME
	# GOTO center slowly
	info('reset')
	data_state = getState()
	x=data_state[0]
	#s.sendall(('0,0,0,0,0,0').encode(FORMAT))#send init state
	reset_time_start=time.time()	
	if x > 0:
		pi.write(16,1); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM) #max 255
		while x>0:                
			time.sleep(0.00001)
			x, _, _,_,_,_ = getState()	
	elif x < 0:
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  rPWM) #max 255		
		while x<0:
			time.sleep(0.00001)
			x, _, _,_,_,_ = getState()
	pi.set_PWM_dutycycle(24,  0)
	initVariables()
	
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
			s.sendall(('0,0,0,0,0,0').encode(FORMAT))
			initVariables()			  
			while True:								
				data_action = s.recv(32).decode(FORMAT)
				info('action: {}'.format(data_action))
				while time.time()-reset_time_start<RESET_TIME:
					time.sleep(0.01)
				if str(data_action=='RESET'):
					reset()					
				else:
					applyForce(float(data_action))
				old_time_chariot = time.time()
				old_time_pendule = time.time()
				time.sleep(0.05)#Te
					
				data_state = getState()				
				data_send = ','.join(str(data_state[i]) for i in range(len(data_state)))				
				s.sendall(data_send.encode(FORMAT))
				
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
