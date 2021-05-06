import os
#TODO bug with long episode

sac_cartpole50={
    gamma: 0.98,
    lr: 0.001292297868387387,
    batch_size: 2048,
    buffer_size: 10000,
    learning_starts: 0,
    train_freq: 64,
    tau: 0.02,
    log_std_init: -0.9655242596737095,
    net_arch: [256,256]
}

noise_rad={

}