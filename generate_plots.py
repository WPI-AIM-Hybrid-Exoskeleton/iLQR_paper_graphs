import matplotlib.pyplot as plt
from generate_data_set import get_data
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal
import numpy as np
import matplotlib
def gen_traj(file_name=None):
    angles = get_data()
    # traj = [angles["Lhip"], angles["Lknee"], angles["Lankle"], angles["Rhip"], angles["Rknee"], angles["Rankle"]]

    min_length = 5000000
    for arr in angles["Rhip"]:
        min_length = min(min_length, len(arr))
    matplotlib.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(3, 2)
    fig.tight_layout(pad=1.0, h_pad=0.15, w_pad=None, rect=None)
    ax[0, 0].set_title("Left Hip")
    ax[1, 0].set_title("Left Knee")
    ax[2, 0].set_title("Left Ankle")
    ax[0, 1].set_title("Right Hip")
    ax[1, 1].set_title("Right Knee")
    ax[2, 1].set_title("Right Ankle")

    ax[2, 0].set_xlabel("Gait %")
    ax[2, 1].set_xlabel("Gait %")

    ax[0, 0].set_ylabel("Angle(deg)")
    ax[1, 0].set_ylabel("Angle(deg)")
    ax[2, 0].set_ylabel("Angle(deg)")


    for i in range(len(angles["Rhip"])):
        ax[0, 0].plot(np.linspace(0, 100, len(angles["Lhip"][i])), np.deg2rad(angles["Lhip"][i]))
        ax[1, 0].plot(np.linspace(0, 100, len(angles["Lknee"][i])), np.deg2rad(angles["Lknee"][i]))
        ax[2, 0].plot(np.linspace(0, 100, len(angles["Lankle"][i])), np.deg2rad( angles["Lankle"][i]))
        ax[0, 1].plot(np.linspace(0, 100, len(angles["Rhip"][i])), np.deg2rad( angles["Rhip"][i]))
        ax[1, 1].plot(np.linspace(0, 100, len(angles["Rknee"][i])), np.deg2rad(angles["Rknee"][i]))
        ax[2, 1].plot(np.linspace(0, 100, len(angles["Rankle"][i])), np.deg2rad(angles["Rankle"][i]))

    if file_name:
        runner = TPGMMRunner.TPGMMRunner(file_name + ".pickle")
        path = np.array(runner.run())
        ax[0, 0].plot( np.linspace(0, 100, len(path[:, 0] )), np.deg2rad(path[:, 0]), linewidth=4)
        ax[1, 0].plot( np.linspace(0, 100, len(path[:, 1] )), np.deg2rad(path[:, 1]), linewidth=4)
        ax[2, 0].plot( np.linspace(0, 100, len(path[:, 2] )), np.deg2rad(path[:, 2]), linewidth=4)
        ax[0, 1].plot( np.linspace(0, 100, len(path[:, 3] )), np.deg2rad(path[:, 3]), linewidth=4)
        ax[1, 1].plot( np.linspace(0, 100, len(path[:, 4] )), np.deg2rad(path[:, 4]), linewidth=4)
        ax[2, 1].plot( np.linspace(0, 100, len(path[:, 5] )), np.deg2rad(path[:, 5]), linewidth=4)

    plt.show()


def train_model(file_name):
    angles = get_data()
    traj = [angles["Lhip"], angles["Lknee"], angles["Lankle"], angles["Rhip"], angles["Rknee"], angles["Rankle"]]
    trainer = TPGMMTrainer.TPGMMTrainer(demo=traj, file_name=file_name, n_rf=20, dt=0.01, reg=[1e-8, 1e-10, 1e-9, 1e-8, 1e-10, 1e-9],
                                                                                               poly_degree=[20, 20, 25, 20, 20, 25])
    trainer.train()


if __name__ == "__main__":
    train_model("leg")
    gen_traj("leg")
    # train_model("leg")
    # compare_model("leg")
