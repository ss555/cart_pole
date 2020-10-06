import numpy as np
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model

class _actor_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range
        self.n1=400
        self.n2=300
    def model(self):
        state = Input(shape=self.state_dim, dtype='float32') #avant float32
        x = Dense(self.n1, activation='relu',kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(
            state)  #
        x = Dense(self.n2, activation='relu',kernel_initializer=RU(-1/np.sqrt(400),1/np.sqrt(400)))(x)
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)  #
        #out = tf.multiply(out, self.action_bound_range)
        return Model(inputs=state, outputs=out)


class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n1=400
        self.n2=300
    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float32') #avant float32
        state_i = Dense(self.n1, activation='relu',kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        # state_i = Dense(128)(state) #kernel_initializer=RU(-1/np.sqrt(301),1/np.sqrt(301))

        action = Input(shape=(self.action_dim,), name='action_input')
        x = concatenate([state_i, action])
        x = Dense(self.n2, activation='relu',kernel_initializer=RU(-1/np.sqrt(401),1/np.sqrt(401)))(x)
        out = Dense(1, activation='linear')(x)  # ,kernel_initializer=RU(-0.003,0.003) ,,kernel_regularizer=l2(0.001)
        return Model(inputs=[state, action], outputs=out)

    
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)