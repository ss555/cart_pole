'''
def plot_experimental_fitted_time_scaled(filename,fA,fB,fC):
    # dv = -a * vs[i] + b * u + c * np.sign(vs[i])+d* np.sign(u)
    plt.figure(figsize=(20,20))
    fileData=np.genfromtxt(filename, delimiter=',').T
    pwmStart = int(min(abs(fileData[:, 0])))
    pwmEnd = int(max(fileData[:, 0]))
    for i in range(pwmStart, pwmEnd + 10, 10):
        # try:
        localData = fileData[fileData[:, 0] == i, :]
        dt = np.mean(np.diff(localData[:, 1]))
        v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
        a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
        u=[((i) / 255 * 12)]
        time_offset=localData[0,1]
        v_fitted = integrate_acceleration(fA,fB,fC,fD,u, timeArr=localData[:,1]-time_offset)
        plt.plot(localData[1:-1, 1]-time_offset, v, '.', label=("%.1fV"%u[0]))
        plt.plot(localData[:,1]-time_offset, v_fitted)
        plt.plot()
        # except:
        #     print('plot_experimental_fitted error')
    for i in range(pwmStart, pwmEnd + 10, 10):
        try:
            localData = fileData[fileData[:, 0] == -i, :]
            dt = np.mean(np.diff(localData[:, 1]))
            v = np.convolve(localData[:, 2], [0.5, 0, -0.5], 'valid') / dt
            a = np.convolve(localData[:, 2], [1, -2, 1], 'valid') / (dt ** 2)
            u=[((-i) / 255 * 12)]
            time_offset = localData[0, 1]
            v_fitted = integrate_acceleration(fA, fB, fC, fD, u, timeArr=localData[:, 1]-time_offset)
            plt.plot(localData[1:-1, 1]-time_offset, v, '.', label=("%.1fV"%u[0]))
            plt.plot(localData[:, 1]-time_offset, v_fitted)
            plt.plot()
        except:
            print('plot_experimental_fitted error')
        plt.plot()
        plt.plot()
        plt.plot()
    plt.legend()
    plt.xlabel("Time, s")
    plt.ylabel("Velocity, m/s")
    plt.grid(True)
    plt.show()
'''