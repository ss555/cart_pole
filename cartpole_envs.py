'''





from filterpy.kalman import JulierSigmaPoints
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
class CartPoleDiscreteKalman(gym.Env):
    def __init__(self,
                 Te=0.05,
                 discreteActions=True,
                 resetMode='random',
                 f_a=A,
                 f_b=B,
                 f_c=C,
                 f_d=D,
                 integrator="semi-euler",
                 tensionMax=8.4706,
                 FILTER=False,
                 n=10,
                 Kp=0,
                 sparseReward=False,
                 Km=0,#1.2,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.sparseReward=sparseReward
        self.Km=Km#bruit du mesure
        self.Kp=Kp#bruit du process
        self.g = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.4611167818372032  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.discreteActions=discreteActions
        if self.discreteActions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space= spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1  # 05
        self.wAngularIni = wAngular  # 4.488 #T=1.4285, w=
        self.reward = None
        self.resetMode = resetMode
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.fd=f_d
        self.tensionMax=tensionMax
        self.arr=[]
        self.n=n
        self.Kp=Kp
        self.FILTER=FILTER
        if self.FILTER:
            self.kfFilter = iir_filter.IIR_filter(signal.butter(4, 0.9, 'lowpass', output='sos'))#2nd param 0.3
            self.kfFilter.R *= .5
            self.kfFilter.Q = Q_discrete_white_noise(2, dt=1., var=0.03)
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        f=self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action[0]+self.fc*np.sign(self.state[1])+self.fd)  #PWM 180 : 7.437548494321268
        return f
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        if self.discreteActions:
            if action==0:
                action=[-1.0]
            elif action==1:
                action=[0.0]
            elif action==2:
                action=[1.0]
            else:
                raise Exception
        else:#continous
            pass
        if self.kinematics_integrator=='semi-euler':
            for i in range(self.n):
                force = self._calculate_force(action)
                # force=np.random.normal(force,)
                xacc = (force + np.random.normal(0,scale=1.0) + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (
                                   self.masscart + self.masspole * sintheta ** 2)  # self.wAngularIni = np.random.normal(wAngular, 0.006, 1)[0]
                thetaacc = self.wAngularIni ** 2 * sintheta + xacc / self.length * costheta - theta_dot * K2
                # xacc=np.random.normal(xacc, 0.03, 1)[0]
                # thetaacc=np.random.normal(thetaacc, 0.03, 1)[0]
                x_dot += self.tau / self.n * xacc
                x += x_dot * self.tau / self.n
                theta_dot += thetaacc * self.tau / self.n
                theta = math.atan2(sintheta, costheta)
                theta += theta_dot * self.tau / self.n
                costheta = np.cos(theta)
                sintheta = np.sin(theta)
        else:
            [x, x_dot, theta, theta_dot] = odeint(self.pend, [x, x_dot, math.atan2(sintheta, costheta), theta_dot], [0, 0.05],
                           args=(action, self.fa, self.fb, self.fc))[-1,:]
        #adding process noise (on x is negligible)
        if self.Kp!=0:
            theta_dot = np.random.normal(theta_dot, self.Kp * 0.2, 1)[0]
            x_dot = np.random.normal(x_dot, self.Kp * 1.2e-3, 1)[0]
            theta = np.random.normal(theta, 0.5 * self.Kp * 0.01, 1)[0]

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        self.state = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta,sintheta=sintheta,theta_dot=theta_dot,sparse=self.sparseReward)
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        # # #adding noise on observed variables
        if self.Km!=0:
            print('noise')
            theta= np.random.normal(theta, 1.5*self.Km * 0.01, 1)[0]
            theta_dot = np.random.normal(theta_dot, self.Km * 0.2, 1)[0]
            x_dot = np.random.normal(x_dot, self.Km * 1.2e-3, 1)[0]
        # # # filtering
        if self.FILTER:
            x_dot = self.iirX_dot.filter(x_dot)
            theta_dot = self.iirTheta_dot.filter(theta_dot)
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}

    def pend(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        dqdt = np.zeros_like(state)
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        dqdt[0] = state[1]
        force=self._calculate_force(action)#0.44*(fa*x_dot+fb*self.tensionMax*action[0]+fc*np.sign(x_dot))
        dqdt[1] = (self.masscart*(fa*x_dot+fb*8.47*action[0]+fc*np.sign(x_dot))#force
                   + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) \
                  / (self.masscart + self.masspole * sintheta ** 2)
        dqdt[3] = self.g / self.length * sintheta# + dqdt[1] / self.length * costheta - theta_dot * K2
        dqdt[2] = state[3]
        # print(dqdt)
        return dqdt
    def reset(self, costheta=-1, sintheta=0, xIni=None,iniSpeed=0.0):
        if self.FILTER:
            # self.iirX_dot=iir_filter.IIR_filter(signal.butter(4, 0.5, 'lowpass', output='sos'))
            self.iirTheta_dot.reset()
        self.COUNTER=0
        self.steps_beyond_done = None
        if self.resetMode=='experimental':
            self.state = np.zeros(shape=(5,))
            if xIni!=None:
                self.state[0] = xIni
                print(f'x = {self.state[0]}')
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        elif self.resetMode=='goal':
            self.state = np.zeros(shape=(5,))
            self.state[2] = 1
            self.state[0] = self.np_random.uniform(low=-0.1, high=0.1)
        elif self.resetMode == 'random':
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[1] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[4] = self.np_random.uniform(low=-5, high=5)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        else:
            print('not defined, choose from experimental/goal/random')
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):

        if self.viewer:
            self.viewer.close()
            self.viewer = None

'''
class CartPoleTime(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print(self.state)
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)

        if self.kinematics_integrator == 'euler':
            xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass, 0)
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = self.rescale_angle(math.pi - self.state[2])
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
'''
class CartPoleDiscrete5(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 Te=0.05,
                 randomReset=False,
                 f_a=-18.03005925191054,
                 f_b=0.965036433340654,
                 f_c=-0.8992003750802359,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.g = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.47  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1  # 05
        self.wAngularIni = 4.488  # 4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset = randomReset
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.tensionMax=8.4706
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        return self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action[0]+self.fc*np.sign(self.state[1]))
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        if action==0:
            action=[-1.0]
        elif action==1:
            action=[-0.5]
        elif action==2:
            action=[0.0]
        elif action==3:
            action=[0.5]
        elif action==4:
            action=[1.0]
        else:
            raise Exception
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = (self._calculate_force(action) + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
                thetaacc = self.g / self.length * sintheta + xacc / self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta   +=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            # theta_dot=np.random.normal(theta_dot,0.2,1)[0]
            # x_dot = np.random.normal(x_dot, 0.1, 1)[0]
        else:
            pass
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)
        # print(f'{self.state}')
        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}
    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

DEBUG=False
#Linearised model
class CartPoleCosSinTensionD3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 randomReset=False,
                 seed = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.46  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians =  math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        # self.v_max = 15
        # self.w_max = 30
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1#05
        self.wAngularIni = 4.6124220495309425#4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset=randomReset
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def gTension(self, u, x_dot=0, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        # return u*slope
    def step(self, action):#180 = 8.47V
        assert self.observation_space.contains(self.state), 'obs_err'
        self.COUNTER+=1
        if action==0:
            action=[-1.0]
        elif action==1:
            action=[0.0]
        elif action==2:
            action=[1.0]
        else:
            raise Exception
        x, x_dot, costheta, sintheta, theta_dot = self.state
        # self.wAngular = np.random.normal(self.wAngularIni,abs(self.wAngularIni/10),1)[0]
        self.wAngular=self.wAngularIni
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(u=action[0] * 8.47,x_dot=x_dot))#- np.sign(x_dot)*0.0438725) # static friction 0.0438725
                # xacc=np.random.normal(xacc,abs(xacc/20),1)[0]
                thetaacc = self.wAngular ** 2 * sintheta + xacc/self.length * costheta #- theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                ## perturbations WORKING
                # x=np.random.normal(x,abs(x/20),1)[0]
                theta = np.random.normal(theta, abs(1*math.pi / 180), 1)[0]
                costheta=np.cos(theta)
                sintheta=np.sin(theta)


            # x_dot=np.random.normal(x_dot,abs(x_dot/10),1)[0]
            theta_dot=np.random.normal(theta_dot,0.2,1)[0]
            x_dot = np.random.normal(x_dot, 0.1, 1)[0]
            # theta_dot = np.random.normal(theta_dot, abs(theta_dot / 30), 1)[0]
        elif self.kinematics_integrator=='rk2':
            xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr

class CartPoleCosSinTensionD(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.48  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 300
        # self.v_max = 15
        # self.w_max = 30
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1
        self.wAngular = 4.4#4.488 #T=1.4285, w=
        self.reward = None
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def gTension(self, u, x_dot=0, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        # return u*slope
    def step(self, action):#180 = 8.47V
        assert self.observation_space.contains(self.state), 'obs_err'
        self.COUNTER+=1
        if action==0:
            action=[-1.0]
        elif action==1:
        #     action=[0.0]
        # elif action==2:
            action=[1.0]
        else:
            raise Exception
        x, x_dot, costheta, sintheta, theta_dot = self.state
        # x_dot=
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(u=action[0] * 8.47,x_dot=x_dot))#- np.sign(x_dot)*0.0438725) # static friction 0.0438725
                thetaacc = self.wAngular ** 2 * sintheta + xacc/self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta+=theta_dot*self.tau/n
                theta = math.atan2(sintheta, costheta)
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            ## perturbations WORKING
            x_dot=np.random.normal(x_dot,abs(x_dot/20),1)
            theta_dot=np.random.normal(theta_dot,abs(theta_dot/20),1)
            # x_dot = np.random.normal(x_dot, abs(x_dot / 30), 1)
            # theta_dot = np.random.normal(theta_dot, abs(theta_dot / 30), 1)
        elif self.kinematics_integrator=='rk2':
            pass
            #xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0, deterministic=True):
        self.COUNTER=0
        self.steps_beyond_done = None
        if deterministic:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr

class CartPoleCosSinTension(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.46  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 3
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.05
        self.wAngular = 4.4#4.488 1.4349472355174064
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def gTension(self, u, x_dot=0, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        # return u*slope

    def step(self, action):#180 = 8.47V
        if not self.observation_space.contains(self.state):
            print(f'obs_err{self.state}')
            raise Exception
        self.COUNTER+=1
        x, x_dot, costheta, sintheta, theta_dot = self.state
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                #xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725) # static friction 0.0438725
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47,x_dot=x_dot))# - np.sign(x_dot)*0.0438725) # static friction 0.0438725
                thetaacc = self.wAngular ** 2 * sintheta + xacc/self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
        #perturbations
        x_dot=np.random.normal(x_dot,abs(x_dot/20),1)
        theta_dot=np.random.normal(theta_dot,abs(theta_dot/20),1)

        # elif self.kinematics_integrator=='rk2':
        #     xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniCartSpeed=0.0,iniThetaSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(5,))
        self.state[1] = iniCartSpeed
        self.state[4] = iniThetaSpeed
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr


#with U/force
class CartPoleCosSinDev(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # center of mass
        self.length = 0.45  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]

        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if DEBUG:
            print(self.state)
            print(action[0])
        assert self.observation_space.contains(self.state), 'obs_err'

        action[0]=_action_static_friction(action[0])
        self.COUNTER += 1
        appliedMaxForce = self.force_mag * self.total_mass
        force = action[0] * appliedMaxForce
        n=2
        for i in range(n):
            x, x_dot, costheta, sintheta, theta_dot = self.state
            theta = math.atan2(sintheta, costheta)
            #TODO!temp = (force + self.polemass_length * theta_dot ** 2 * sintheta - np.sign(x_dot)*appliedMaxForce*0.08)/self.total_mass
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot

            if self.kinematics_integrator == 'euler':
                xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            elif self.kinematics_integrator == 'friction':
                xacc = -K1 * x_dot + temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            theta = self.rescale_angle(theta)
            self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
            done = False
            if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
                done = True
                x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        # cost = reward_fnCos(x, costheta, x_dot)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(5,))#
        # self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
'''
class CartPoleBullet(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print(self.state)
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)

        if self.kinematics_integrator == 'euler':
            xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass, 0)
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<=-math.pi:
                theta+=2*math.pi
            theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)

        cost=reward_fn(x, theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER = 0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[2] = -1
        self.state[3] = 0
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleCosSinRPIhistory(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 pi_conn,
                 k_history_len=2,
                 seed: int = 0,):
        self.MAX_STEPS_PER_EPISODE = 400
        self.FORMAT = 'utf-8'
        self.counter=0
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        self.v_max = 100
        self.w_max = 100
        self.k_history_len = k_history_len
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            1.0,
            1.0,
            self.w_max])
        high = np.append(high, np.ones(shape=self.k_history_len))
        self.action_history_buffer = deque(np.zeros(self.k_history_len), maxlen=self.k_history_len)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype = np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.conn = pi_conn
        print('connected')
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #send action receive data-old
        self.counter+=1
        assert self.observation_space.contains(self.state), 'obs_err{}'.format(self.state)
        self.conn.sendall(str(action[0]).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        done = bool(state[-1])
        self.state=state[:-1]
        self.action_history_buffer.append(action[0])

        self.state=np.append(self.state,self.action_history_buffer)
        self.state[2]=np.clip(self.state[2],-1,1)
        self.state[3]=np.clip(self.state[3],-1,1)
        x = self.state[0]
        costheta = self.state[2]
        cost = reward_fnCosSin(x, costheta, state[4])

        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 5
            print('out of bound')
            self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
        if self.MAX_STEPS_PER_EPISODE==self.counter:
            done=True
        info('state: {}, cost{}, done:{}'.format(self.state,cost,done))

        return self.state, cost, done, {}
    def reset(self):

        self.conn.sendall('RESET'.encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        self.state = state[:-1]
        info('reset with nsteps: {}, state:{}'.format(self.counter, self.state))
        self.counter = 0
        self.state=np.append(self.state,np.zeros(self.k_history_len))
        print('state {}'.format(self.state))
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
class CartPoleCosSinWorking(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]

        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        # action=_action_static_friction(action)
        self.COUNTER += 1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, costheta, sintheta, theta_dot = self.state
        if DEBUG:
            print(self.state)
        force = action[0] * self.force_mag
        theta = math.atan2(sintheta, costheta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        theta = self.rescale_angle(theta)
        self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        done = False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleCosSinHistory(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 k_history_len=2,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # center of mass
        self.length = 0.47  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]
        self.k_history_len = k_history_len
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        high=np.append(high,np.ones(shape=(k_history_len,)))
        self.action_history_buffer = deque(np.zeros(self.k_history_len), maxlen=self.k_history_len)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if DEBUG:
            print(self.state)
            print(action[0])
        assert self.observation_space.contains(self.state), 'obs_err'
        self.action_history_buffer.append(action[0])
        action[0]=_action_static_friction(action[0])
        self.COUNTER += 1
        force = action[0] * self.force_mag * self.total_mass

        n=2
        for i in range(n):

            theta = math.atan2(sintheta, costheta)
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot

            if self.kinematics_integrator == 'euler':
                xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            elif self.kinematics_integrator == 'friction':
                xacc = -K1 * x_dot + temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            theta = self.rescale_angle(theta)
            self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
            done = False
            if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
                done = True
                x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        self.state=np.append(self.state,self.action_history_buffer)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(7,))#
        # self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
class CartPoleCusBottomDiscrete(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=N_STEPS
        self.gravity = 10
        self.masscart = 0.45
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 0.2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.K1 = K1 / self.total_mass
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot = state
        if action == 0:
            force = 0
        elif action == 1:
            force = -self.force_mag
        elif action == 2:
            force = self.force_mag
        else:
            print('invalid action space asked')
            exit(-1)
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = self.K1 * x_dot  + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True

        #cost = reward_fn(x,x_dot,theta,theta_dot,self.x_threshold,self.observation_space.high[1],self.observation_space.high[-1],action)
        cost=reward_fn(x,theta)
        if x <= -self.x_threshold or x >= self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}
#tensorboard --logdir ./sac_cartpole_tensorboard/
    #INFO
    #reward_fn 4096 takes 1h to train
    #sparse rewards doesnt learn
    def reset(self):
        # self.state = np.array([0, 0, math.pi, 0]).astype(np.float32)
        self.steps_beyond_done = None
        self.COUNTER = 0
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi - self.state[2]
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleDiscrete2actions(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=1000
        self.gravity = 10
        self.masscart = 0.45
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 0.05
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max,
            1])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.COUNTER += 1
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot, _ = state
        if action == 0:
            force = self.force_mag
        elif action == 1:
            force = -self.force_mag
        else:
            print('invalid action space asked')
            exit(-1)
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot, self.COUNTER/self.MAX_STEPS_PER_EPISODE],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}
    def reset(self):
        self.steps_beyond_done = None
        self.COUNTER = 0
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi - self.state[2]
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        self.state=np.append(self.state,0)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleConTest(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 time_limit=400,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE = time_limit
        self.gravity = 10
        self.masscart = 0.45
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max,
            1])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot, _ = state
        force = action*self.force_mag
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.MAX_STEPS_PER_EPISODE == self.COUNTER:
            done = True

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        state[2] = math.pi - state[2]
        state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0,shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        action=_action_static_friction(action)
        x, x_dot, theta, theta_dot = self.state
        force = action# self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        print('force app', force)
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        print('state is',self.state)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[1] = 0
        self.state[2]=math.pi
        self.state[3]=3
        self.steps_beyond_done = None
        print('reset')
        print(self.state)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleCusBottom(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            self.theta_threshold_radians,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, theta, theta_dot = self.state
        force = action*self.force_mag
        # print(force)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        theta=self.rescale_angle(theta)
        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        # print('reset state:',self.state)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleCosSinFriction(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 k_history_len = 2,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.k_history_len = k_history_len
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        high=np.append(high,np.ones(shape=self.k_history_len))
        self.action_history_buffer=deque(np.zeros(self.k_history_len), maxlen=self.k_history_len)
        # self.init_buffer(self.action_history_buffer)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)

        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        force = action[0] * self.force_mag
        x, x_dot, costheta, sintheta, theta_dot = self.state[:5]

        theta=math.atan2(sintheta, costheta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) -K2*theta_dot
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass #adding -0.1 * x_dot / self.total_mass
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        theta=self.rescale_angle(theta)
        self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot],dtype=np.float32)
        self.state = np.append(self.state, self.action_history_buffer)
            # print('{}'.format(self.state))
        self.action_history_buffer.append(action[0])
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost=reward_fnCos(x, costheta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/5
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(self.observation_space.shape[0]))#self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[2] = -1
        self.state[3] = 0
        # self.state[0] = self.np_random.uniform(low=-2, high=2)
        # print('reset state:',self.state)
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



class CartPoleRPI(gym.Env):
                metadata = {
                    'render.modes': ['human', 'rgb_array'],
                    'video.frames_per_second': 50
                }

                def __init__(self,
                             pi_conn,
                             k_history_len=2,
                             seed: int = 0, ):
                    self.MAX_STEPS_PER_EPISODE = 3000
                    self.FORMAT = 'utf-8'

                    # Angle at which to fail the episode
                    self.theta_threshold_radians = 180 * 2 * math.pi / 360
                    self.x_threshold = 5.0
                    self.v_max = 100
                    self.w_max = 100
                    self.k_history_len = k_history_len
                    # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
                    high = np.array([
                        self.x_threshold,
                        self.v_max,
                        self.theta_threshold_radians,
                        self.w_max])
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                    self.observation_space = spaces.Box(-high, high, dtype=np.float32)
                    self.seed(seed)
                    self.viewer = None
                    self.state = None
                    self.steps_beyond_done = None
                    self.conn = pi_conn

                    print('connected')

                def seed(self, seed=0):
                    self.np_random, seed = seeding.np_random(seed)
                    return [seed]

                def step(self, action):
                    # send action receive data-old
                    assert self.observation_space.contains(self.state), 'obs_err'
                    self.conn.sendall(str(action[0]).encode(self.FORMAT))
                    sData = self.conn.recv(124).decode(self.FORMAT)
                    state = np.array(sData.split(',')).astype(np.float32)
                    done = bool(state[4])
                    self.state = state[:-1]

                    x = self.state[0]
                    theta = self.state[2]
                    cost = reward_fn(x, theta)
                    if x < -self.x_threshold or x > self.x_threshold:
                        cost = cost - 100
                        self.state[0] = np.clip(x, -self.x_threshold, self.x_threshold)
                    print(self.state)
                    # print('done',done)
                    return self.state, cost, done, {}

                def reset(self):
                    self.conn.sendall('RESET'.encode(self.FORMAT))
                    print('reset')
                    sData = self.conn.recv(124).decode(self.FORMAT)
                    state = np.array(sData.split(',')).astype(np.float32)
                    self.state = state[:-1]
                    return self.state

                def render(self, mode='human'):
                    pass

                def close(self):
                    if self.viewer:
                        self.viewer.close()
                        self.viewer = None