import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal

import numpy as np
from GaitAnaylsisToolkit.Session import ViconGaitingTrial


def get_joint_angles(files, indecies, sides):
    """

    :param files:
    :param indecies:
    :param sides:
    :param lables:
    :return:
    """
    angles = {}
    angles["hip"] = []
    angles["knee"] = []
    angles["ankle"] = []

    angles2 = {}
    angles2["Rhip"] = []
    angles2["Rknee"] = []
    angles2["Rankle"] = []
    angles2["Lhip"] = []
    angles2["Lknee"] = []
    angles2["Lankle"] = []

    samples = []
    for file, i, side in zip(files, indecies, sides):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        body = trial.get_joint_trajectories()
        if side == "L":
            hip_angle = body.left.hip[i].angle.x.data
            knee_angle = body.left.knee[i].angle.x.data
            ankle_angle = body.left.ankle[i].angle.x.data
        else:
            hip_angle = body.right.hip[i].angle.x.data
            knee_angle = body.right.knee[i].angle.x.data
            ankle_angle = body.right.ankle[i].angle.x.data

        angles["hip"].append(hip_angle)
        angles["knee"].append(knee_angle)
        angles["ankle"].append(ankle_angle)
        samples.append(len(hip_angle))

    sample_size = min(samples)

    # for i in range(len(files)):
    #     angles2["hip"].append(signal.resample(angles["hip"][i], sample_size))
    #     angles2["knee"].append(signal.resample(angles["knee"][i], sample_size))
    #     angles2["ankle"].append(signal.resample(angles["ankle"][i], sample_size))

    for i in range(len(files)):
        angles2["Lhip"].append(angles["hip"][i])
        angles2["Lknee"].append(angles["knee"][i])
        angles2["Lankle"].append(angles["ankle"][i])

        angles2["Rhip"].append(np.flip(angles["hip"][i]))
        angles2["Rknee"].append(np.flip(angles["knee"][i]))
        angles2["Rankle"].append(np.flip(angles["ankle"][i]))

    return angles2



def get_data():
    angles = get_joint_angles(["/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_01/subject_01_walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_04/subject_04_walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 walk_01.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_walking_01.csv",
                               "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv"],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 2],
                              ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R"])
    return angles

#if __name__ == "__main__":
#     # angles = plot_joint_angles(["/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_walk_00.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_04/subject_04_walk_00.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_walk_00.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 walk_01.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_walking_01.csv",
#     #                             "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv",
#     #                             ],
#     #                            [1, 0, 0, 0, 0, 0, 0, 2],
#     #                            ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R"],
#     #                            ["Subject00", "Subject01", "Subject02", 'Subject03', "subject04", "Subject05",
#     #                             "Subject06", "Subject07", "Subject08", "Subject10"])
#
    # angles = get_joint_angles(["/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_01/subject_01_walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_04/subject_04_walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 walk_01.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_walking_01.csv",
    #                    "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv"],
    #                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 2],
    #                   ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R"])
    #
    # traj = [angles["hip"], angles["knee"], angles["ankle"]]
    #
    #
    # trainer = TPGMMTrainer.TPGMMTrainer(demo=traj, file_name="leg", n_rf=15, dt=0.01, reg=[1e-5, 1e-7, 1e-7], poly_degree=[20, 20, 20])
    # trainer.train()
    # runner = TPGMMRunner.TPGMMRunner("leg.pickle")
    # path = np.array(runner.run())
#
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#
#     for p in angles["hip"]:
#         ax1.plot(p)
#     for p in angles["knee"]:
#         ax2.plot(p)
#     for p in angles["ankle"]:
#         ax3.plot(p)
#
#     ax1.plot(path[:, 0], linewidth=4)
#     ax2.plot(path[:, 1], linewidth=4)
#     ax3.plot(path[:, 2], linewidth=4)
#     plt.show()