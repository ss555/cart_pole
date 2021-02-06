#!/usr/bin/env python3

import socket
#'fe80::53e3:b88f:2855:ca06' #ipv6
HOST = '169.254.161.71'#'c0:3e:ba:6c:9e:eb'#'10.42.0.1'#wifiHot #'127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
FORMAT = 'utf-8'
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()

    with conn:
        print('Connected by', addr)
        for i in range(100):
            data = conn.recv(1024)
            if not data:
                break
            data_raw = data.decode(FORMAT)
            print(data_raw)
            action = 0.984546515
            conn.sendall(str(action).encode(FORMAT))
    conn.close()