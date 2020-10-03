import matplotlib.pyplot as plt
from generate_data_set import get_data
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal

def gen_traj():
    angles = get_data()
    # traj = [angles["Lhip"], angles["Lknee"], angles["Lankle"], angles["Rhip"], angles["Rknee"], angles["Rankle"]]

    min_length = 5000000
    for arr in angles["Rhip"]:
        min_length = min(min_length, len(arr))

    fig, ax = plt.subplots(3, 2)

    for i in range(len(angles["Rhip"])):
        ax[0, 0].plot(signal.resample(angles["Lhip"][i], min_length))
        ax[1, 0].plot(signal.resample(angles["Lknee"][i], min_length))
        ax[2, 0].plot(signal.resample(angles["Lankle"][i], min_length))
        ax[0, 1].plot(signal.resample(angles["Rhip"][i], min_length))
        ax[1, 1].plot(signal.resample(angles["Rknee"][i], min_length))
        ax[2, 1].plot(signal.resample(angles["Rankle"][i], min_length))

    plt.show()


def compare_model(file_name):
    runner = TPGMMRunner.TPGMMRunner( file_name + ".pickle")

def train_model(file_name):
    angles = get_data()
    traj = [angles["Lhip"], angles["Lknee"], angles["Lankle"], angles["Rhip"], angles["Rknee"], angles["Rankle"]]
    trainer = TPGMMTrainer.TPGMMTrainer(demo=traj, file_name=file_name, n_rf=15, dt=0.01, reg=[1e-5, 1e-7, 1e-7, 1e-5, 1e-7, 1e-7], poly_degree=[20, 20, 20, 20, 20, 20])
    trainer.train()


if __name__ == "__main__":
    gen_traj()
    # train_model("leg")
    # compare_model("leg")
