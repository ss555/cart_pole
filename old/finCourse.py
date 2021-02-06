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

def cbf(gpio, level, tick):
	pi.set_PWM_dutycycle(24,  0)
	print('capteur fin course');
	global start_time
	print(start_time-time.time())



cb1 = pi.callback(17, pigpio.EITHER_EDGE, cbf)
cb2 = pi.callback(18, pigpio.EITHER_EDGE, cbf)

posPendule = 0
def callbackPendule(way):
  global posPendule
  posPendule += way
  print("pos pendule={}".format(posPendule%600))
  
  
posChariot = 180
def callbackChariot(way):
  global posChariot
  posChariot += way
  print("pos chariot={}".format(posChariot))
  
if __name__ == "__main__":
	try:
		decoderPendule = decoder(pi, 19, 26, callbackPendule) #pendule
		decoderChariot = decoder(pi, 20, 21, callbackChariot) #chariot
		print('power')
		start_time = time.time()
		pi.write(16,0); #1-right 0-left
		pi.set_PWM_dutycycle(24,  10) #max 255
		while 1:
			if posChariot<6250:
				pi.set_PWM_dutycycle(24,  50)
			else:
				pi.set_PWM_dutycycle(24,  0)
				break
		pi.write(16,1);
		pi.set_PWM_dutycycle(24,  0)
		
		decoderPendule.cancel()
		decoderChariot.cancel() 
	except:
		# en cas de CTRL+C ou ERREUR dans le programme mise a zero Pmoteur
		pi.set_PWM_dutycycle(24,  0)
	finally:
		pi.set_PWM_dutycycle(24,  0)


'''

 total length 12500 ticks Chariot, it takes 8.8 sec with PWM=50 to go full length
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
