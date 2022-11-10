
def plot_time_evolution(ts, data_block, save_path_fig):
    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(ts,data_block[:,0], label= 'left')
    plt.plot(ts, data_block[:,1], label= 'above tumor')
    plt.plot(ts,data_block[:,2], label= 'right')
    plt.xlabel("t")
    plt.ylabel('Light intensity')
    plt.legend()
    plt.savefig(save_path_fig)
