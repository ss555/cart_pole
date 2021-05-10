import os
#TODO bug with long episode
dqn_sim50={
    policy: 'MlpPolicy',
    learning_starts:10000,
    gamma:0.995,
    learning_rate:0.0003,
    exploration_final_eps=0.17787752200089602,
    exploration_initial_eps=0.17787752200089602,
    target_update_interval=1000,
    buffer_size=50000,
    train_freq=(1, "episode"),
    gradient_steps=-1,#n_episodes_rollout=1,
    verbose=0,
    batch_size=1024,
    policy_kwargs=dict(net_arch=[256, 256])}
sac_cartpole50={
    gamma: 0.98,
    lr: 0.001292297868387387,
    batch_size: 2048,
    buffer_size: 10000,
    learning_starts: 0,
    train_freq : (1, "episode"),
    tau: 0.02,
    log_std_init: -0.9655242596737095,
    net_arch: [256,256]
}
dqn_20={
    gamma: 0.95,
    lr: 0.0018307754257612685,
    batch_size: 512,
    buffer_size: 1000000,
    exploration_final_eps: 0.02633362231811897,
    exploration_fraction: 0.22175283117941602,
    target_update_interval: 1,
    learning_starts: 10000,
    train_freq: 1000,
    subsample_steps: 2,
    net_arch: [64,64]
}
noise_rad={

}