#!/usr/bin/env python3
import zmq
import time


## Python class that provides a bridge to the low-level interface.
# This Python class allows one to communicate with the low-level interface
# using two TCP sockets (one to receive the current state of the pendulum,
# one to send actuation commands).
class PendulePy:
  ## Number of state coordinates.
  N_STATES = 5
  ## Constructor, initializes socket connections.
  # Connections are established at `tcp://[host]:[port]`. The
  # @param host string that tells the host for the socket.
  # @param state_port port of the socket that is used to read the current
  #   state of the pendulum.
  # @param command_port port of the socket that is used to send commands to
  #   the low-level interface.
  # @param wait time to wait for the interface, in seconds. If less than or
  #   equal to zero, wait indefinitely. If positive, the constructor will check
  #   if a message is available every second, until wait seconds have elapsed.
  #   If no message is received within the allotted time, an exception will be
  #   thrown.
  def __init__(self, host="localhost", state_port="10001", command_port="10002", wait=-1):
    # ZeroMQ entrypoint.
    self._context = zmq.Context()
    # Socket to send commands to the low-level interface.
    self._command_pub = self._context.socket(zmq.PUB)
    self._command_pub.connect(f"tcp://{host}:{command_port}")
    # Socket to receive the current state from the low-level interface.
    self._state_sub = self._context.socket(zmq.SUB)
    self._state_sub.set(zmq.CONFLATE, 1)
    self._state_sub.connect(f"tcp://{host}:{state_port}")
    self._state_sub.subscribe("")
    # Initialize the state
    self._time = None
    self._position = None
    self._angle = None
    self._linvel = None
    self._angvel = None
    # Wait for the low-level interface to be up and running.
    if wait <= 0:
      # Wait indefinitely for the first message.
      self.readState(blocking=True)
    else:
      # Wait for the given amount of time.
      elapsed = 0
      while elapsed < wait:
        # Wait one second.
        time.sleep(1)
        # Check if it is possible to read a message.
        if self.readState(blocking=False):
          # A message was received: exit!
          break
        elapsed += 1
      # Did we exit due to a received message? If not, throw!
      if elapsed >= wait:
        raise RuntimeError("PendulePy: failed to establish a connection with the low-level interface within the allotted time.")

  ## Allows to access the current time of the pendulum.
  @property
  def time(self):
    return self._time

  ## Allows to access the current position of the pendulum.
  @property
  def position(self):
    return self._position

  ## Allows to access the current angle of the pendulum.
  @property
  def angle(self):
    return self._angle

  ## Allows to access the current linear velocity of the pendulum.
  @property
  def linvel(self):
    return self._linvel

  ## Allows to access the current angular velocity of the pendulum.
  @property
  def angvel(self):
    return self._angvel

  ## Tries to read the state of the pendulum from the interface.
  # @param blocking if `True`, do not exit until a message has been received from
  #   the socket. If `False`, exit immediately if no messages are received.
  # @return `True` if a message has been received and processed. Note that if
  #   the call is blocking, the function should always return `True`.
  # @note While by default the parameter `blocking` is set to `True`, we
  #   recommend to always pass it as keyword argument, *e.g.*,
  #   `p.readState(blocking=True)`. In this way, it is always clear whether
  #   the call is blocking or not.
  def readState(self, blocking=True):
    if blocking:
      # Wait for a string message from the socket.
      msg = self._state_sub.recv_string()
    else:
      try:
        # Read a string message from the socket - if none is available, throw!
        msg = self._state_sub.recv_string(flags=zmq.NOBLOCK)
      except zmq.Again:
        # No message has been received from the socket.
        return False
    # Split the string message into parts. Each part should be a float.
    parts = msg.split(" ")
    # Check that the number of parts is correct, and throw otherwise.
    if len(parts) != PendulePy.N_STATES:
      raise RuntimeError(f"Malformed state message received. Expected {PendulePy.N_STATES} values, got {len(parts)}. Message: {msg}")
    # Convert each part (which is still a string) into a float.
    self._time, self._position, self._angle, self._linvel, self._angvel = map(float, parts)
    # State read successfully.
    return True

  ## Send a PWM command to the low-level interface.
  # @param pwm the PWM signal to be sent. It should be an integer between -255
  #   and 255. The parameter is cast to `int` before being sent.
  def sendCommand(self, pwm):
    self._command_pub.send_string(str(int(pwm)))
