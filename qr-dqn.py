from sb3_contrib import QRDQN
import gym
policy_kwargs = dict(n_quantiles=50)
env = gym.make("CartPole-v1")
model = QRDQN("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("qrdqn_cartpole")
obs=env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()