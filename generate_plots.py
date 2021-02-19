import matplotlib.pyplot as plt
from generate_data_set import get_data
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
from dtw import dtw
import numpy.polynomial.polynomial as poly

def gen_traj(file_name=None):
    angles = get_data()
    # traj = [angles["Lhip"], angles["Lknee"], angles["Lankle"], angles["Rhip"], angles["Rknee"], angles["Rankle"]]

    min_length = 5000000
    for arr in angles["Rhip"]:
        min_length = min(min_length, len(arr))
    matplotlib.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(3)
    fig.tight_layout(pad=1.0, h_pad=0.15, w_pad=None, rect=None)
    ax[0].set_title("Left Hip")
    ax[1].set_title("Left Knee")
    ax[2].set_title("Left Ankle")
    # ax[0, 1].set_title("Right Hip")
    # ax[1, 1].set_title("Right Knee")
    # ax[2, 1].set_title("Right Ankle")

    ax[2].set_xlabel("Gait %")
    #ax[2, 1].set_xlabel("Gait %")

    ax[0].set_ylabel("Angle(deg)")
    ax[1].set_ylabel("Angle(deg)")
    ax[2].set_ylabel("Angle(deg)")


    for i in range(len(angles["Rhip"])):
        ax[0].plot(np.linspace(0, 100, len(angles["Lhip"][i])), np.rad2deg(angles["Lhip"][i]))
        ax[1].plot(np.linspace(0, 100, len(angles["Lknee"][i])), np.rad2deg(angles["Lknee"][i]))
        ax[2].plot(np.linspace(0, 100, len(angles["Lankle"][i])), np.rad2deg( angles["Lankle"][i]))
        # ax[0, 1].plot(np.linspace(0, 100, len(angles["Rhip"][i])), np.rad2deg(angles["Rhip"][i]))
        # ax[1, 1].plot(np.linspace(0, 100, len(angles["Rknee"][i])), np.rad2deg(angles["Rknee"][i]))
        # ax[2, 1].plot(np.linspace(0, 100, len(angles["Rankle"][i])), np.rad2deg(angles["Rankle"][i]))

    if file_name:
        runner = TPGMMRunner.TPGMMRunner(file_name + ".pickle")
        path = np.array(runner.run())
        ax[0].plot( np.linspace(0, 100, len(path[:, 0] )), np.rad2deg(path[:, 0]), linewidth=4)
        ax[1].plot( np.linspace(0, 100, len(path[:, 1] )), np.rad2deg(path[:, 1]), linewidth=4)
        ax[2].plot( np.linspace(0, 100, len(path[:, 2] )), np.rad2deg(path[:, 2]), linewidth=4)
    #     ax[0, 1].plot( np.linspace(0, 100, len(path[:, 3] )), np.rad2deg(path[:, 3]), linewidth=2)
    #     ax[1, 1].plot( np.linspace(0, 100, len(path[:, 4] )), np.rad2deg(path[:, 4]), linewidth=2)
    #     ax[2, 1].plot( np.linspace(0, 100, len(path[:, 5] )), np.rad2deg(path[:, 5]), linewidth=2)
    #
    plt.show()




def plot_gmm(Mu, Sigma, ax=None):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    X = []
    nb_state = len(Mu[0])
    patches = []

    for i in range(nb_state):
        w, v = np.linalg.eig(Sigma[i])
        R = np.real(v.dot(np.lib.scimath.sqrt(np.diag(w))))
        x = R.dot(np.array([np.cos(t), np.sin(t)])) + np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, nbDrawingSeg)
        x = x.transpose().tolist()
        patches.append(Polygon(x, edgecolor='r'))
        ax.plot(Mu[0, i], Mu[1, i], 'm*', linewidth=10)

    p = PatchCollection(patches, edgecolor='k', color='green', alpha=0.8)
    ax.add_collection(p)

    return p

def get_gmm(file_name):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}

    matplotlib.rc('font', **font)

    nb_states = 10

    runner = GMMRunner.GMMRunner(file_name)

    fig0, ax = plt.subplots(3,sharex=True)

    sIn = runner.get_sIn()
    tau = runner.get_tau()
    l = runner.get_length()
    motion = runner.get_motion()
    mu = runner.get_mu()
    sigma = runner.get_sigma()
    currF = runner.get_expData()

    # plot the forcing functions
    angles = get_data()
    for i in range(len(angles["Lhip"])):
        ax[0].plot(sIn, tau[1, i * l: (i + 1) * l].tolist(), color="b")
        ax[1].plot(sIn, tau[2, i * l: (i + 1) * l].tolist(), color="b")
        ax[2].plot(sIn, tau[3, i * l: (i + 1) * l].tolist(), color="b")

    ax[0].plot(sIn, currF[0].tolist(), color="y", linewidth=5)
    ax[1].plot(sIn, currF[1].tolist(), color="y", linewidth=5)
    ax[2].plot(sIn, currF[2].tolist(), color="y", linewidth=5)

    sigma0 = sigma[:, :2, :2]
    sigma1 = sigma[:, :3, :2]
    sigma2 = sigma[:, :4, :2]

    sigma1 = np.delete(sigma1, 1, axis=1)
    sigma2 = np.delete(sigma2, 1, axis=1)
    sigma2 = np.delete(sigma2, 1, axis=1)

    p = plot_gmm(Mu=np.array([mu[0,:], mu[1,:] ]), Sigma=sigma0, ax=ax[0])
    p = plot_gmm(Mu=np.array([mu[0, :], mu[2, :]]), Sigma=sigma1, ax=ax[1])
    p = plot_gmm(Mu=np.array([mu[0, :], mu[3, :]]), Sigma=sigma2, ax=ax[2])
    fig0.suptitle('Forcing Function')

    ax[2].set_xlabel('S')
    ax[0].set_ylabel('F')
    ax[1].set_ylabel('F')
    ax[2].set_ylabel('F')

    # fig0.tight_layout(pad=1.0, h_pad=0.15, w_pad=None, rect=None)
    ax[0].set_title("Left Hip")
    ax[1].set_title("Left Knee")
    ax[2].set_title("Left Ankle")

    plt.show()


def train_model(file_name, bins=15, save=True):
    angles = get_data()
    traj = [angles["Lhip"], angles["Lknee"], angles["Lankle"], angles["Rhip"], angles["Rknee"], angles["Rankle"]]
    trainer = TPGMMTrainer.TPGMMTrainer(demo=traj, file_name=file_name, n_rf=bins, dt=0.01, reg=[1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
                                                                                               poly_degree=[20, 20, 20, 20, 20, 20])
    return trainer.train(save)


def get_BIC(file_name):

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}

    matplotlib.rc('font', **font)

    for j in range(30):
        BIC = {}
        for i in range(5,30):
            data = train_model(file_name, bins=i, save=False)
            BIC[i] = data["BIC"]

        plt.plot(list(BIC.keys()), list(BIC.values()))

    plt.xlabel("Bins")
    plt.ylabel("BIC")
    plt.title("BIC score for Walking")
    plt.show()


def calculate_imitation_metric(file_name):
    angles = get_data()
    demos = [angles["Lhip"], angles["Lknee"], angles["Lankle"]]
    runner = TPGMMRunner.TPGMMRunner(file_name)
    path = runner.run()
    print(path[:,0])


    alpha = 1.0
    manhattan_distance = lambda x, y: abs(x - y)

    costs = []
    for i in range(3):
        imitation = path[:, i]
        T = len(imitation)
        M = len(demos[i])
        metric = 0.0
        t = []
        t.append(1.0)
        for k in range(1, T):
            t.append(t[k - 1] - alpha * t[k - 1] * 0.01)  # Update of decay term (ds/dt=-alpha s) )
        t = np.array(t)

        for m in range(M):
            d, cost_matrix, acc_cost_matrix, path_im = dtw(imitation, demos[i][m], dist=manhattan_distance)
            data_warp = [demos[i][m][path_im[1]][:imitation.shape[0]]]
            coefs = poly.polyfit(t, data_warp[0], 20)
            ffit = poly.Polynomial(coefs)
            y_fit = ffit(t)
            metric += np.sum(abs(y_fit - imitation.flatten()))

        costs.append(metric / (M * T))
        print("cost ")
        print(costs)
    return costs

if __name__ == "__main__":
    #train_model("walk2")
    #get_BIC("walk2")
    #gen_traj("walk2")
    #get_gmm("walk2.pickle")
    #calculate_imitation_metric("walk2.pickle")
    # train_model("leg")
    # compare_model("leg")
