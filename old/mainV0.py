#!/usr/bin/env python
import time
from rotary_encoder import decoder
#from stable_baselines import SAC
'''todo

'''
import pigpio
PI_INPUT = pigpio.INPUT
PI_PUD_UP = pigpio.PUD_UP
pi = pigpio.pi()
if not pi.connected:
    exit()
    
    
#Fin de course
pi.set_mode(17, PI_INPUT); #//FDC17 
pi.set_mode(18, PI_INPUT); #//FDC18
pi.set_pull_up_down(17, PI_PUD_UP);   
pi.set_pull_up_down(18, PI_PUD_UP); 

oldIFinCourseMoteur = pi.read(18);	#//1 si pas appui
oldIFinCourseEncodeur = pi.read(17);  #//1 si pas appui
print(oldIFinCourseMoteur)
print(oldIFinCourseEncodeur)
STOP = 0
def cbf(gpio, level, tick):	
	pi.set_PWM_dutycycle(24,  0)
	global STOP
	STOP = 1

cb1 = pi.callback(17, pigpio.EITHER_EDGE, cbf)
cb2 = pi.callback(18, pigpio.EITHER_EDGE, cbf)

posPendule = 300
def callbackPendule(way):
	global posPendule
	posPendule += way
	print("pos pendule={}".format((posPendule%600)*0.3))#*180/600
  
  
posChariot = 0
def callbackChariot(way):
	global posChariot
	posChariot += way
	print("pos chariot={}".format(posChariot))
	
	
def goCenter(position):
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
def oscillations(amplitude,n_repetitions=5):
	global position
	for i in range(n_repetions):
		while position<amplitude:
			pi.write(16,0); #1-right 0-left
			pi.set_PWM_dutycycle(24,  150)
		while position>-amplitude:
			pi.write(16,1); #1-right 0-left
			pi.set_PWM_dutycycle(24,  150)
if __name__ == "__main__":
	try:
		decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
		decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
		# sens encodeur +6250:left 0-middle -6250: right
		print('power')
		start_time = time.time()
		
		#HYPOTHESE!!!: start in middle
		print('posChariot initiale: ',posChariot)
		while 1:
			#main			
			#if posChariot>-5500: # right
			if posChariot<5500: #left
				pi.write(16,0); #1-right 0-left
				pi.set_PWM_dutycycle(24,  100)
			else:
				pi.set_PWM_dutycycle(24,  0)
				break
			if STOP:
				print('arret d\'urgence')
				pi.set_PWM_dutycycle(24,  0)
				break
		
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  0)
		
		decoderPendule.cancel()
		decoderChariot.cancel() 
	except:
		print('smth went wrong')
		# en cas de CTRL+C ou ERREUR dans le programme mise a zero Pmoteur
		pi.set_PWM_dutycycle(24,  0)


print('total program time: ',start_time-time.time())
'''

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
'''

'''
if posChariot>6250:
		pi.set_PWM_dutycycle(24,  0)
		break
		
while 1:
	if abs(posChariot)>5500:
		goCenter(posChariot)
		break
'''
