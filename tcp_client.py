#!/usr/bin/env python3
#wifi latency aller-retour can go up to 0.1s, alors que par fil: moins de 1ms
import socket
import time
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifi-hotspot  # The server's hostname or IP address
PORT = 65432        # The port used by the server
FORMAT = 'utf-8'
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.connect((HOST, PORT))
	for i in range(100):		
		start_time=time.time()
		states=[1.6368261287223835,1.6368261287223835,1.6368261287223835,1.6368261287223835,0]
		data=','.join(str(states[i]) for i in range(len(states)))
		s.sendall(data.encode(FORMAT))
		time.sleep(0.02)
		action = s.recv(16).decode(FORMAT)
		print(float(action))
		print(time.time()-start_time)#
print('Received', repr(data))

