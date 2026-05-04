import numpy as np
from dmp.dmp import DMPModel
import os
from dmp.dmp import rollout_simple_with_coupling
from mapping.retarget import JOINT_LIMITS_DEG

def get_trajectories(subject: int = 10):
    """
    Get the trajectories for the experiment.
    """

    demo = np.array([[ 57.697568,  -0.829577,   2.918262,  -6.047991],
            [ 57.138965,  -1.284988,   2.797236,  -6.488143],
            [ 55.771160,  -1.939663,   2.587255,  -7.337363],
            [ 53.571704,  -2.610220,   2.324964,  -8.535971],
            [ 50.647542,  -3.126743,   2.049133,  -9.999632],
            [ 47.119936,  -3.246150,   1.806035, -11.578808],
            [ 43.042074,  -2.659116,   1.649260, -13.060559],
            [ 38.472927,  -1.248614,   1.617750, -14.243157],
            [ 33.600655,   1.355072,   1.757217, -14.941738],
            [ 28.749955,   5.590941,   2.103592, -15.067616],
            [ 24.239225,  11.785120,   2.669339, -14.653310],
            [ 20.453589,  19.888113,   3.438150, -13.824522],
            [ 17.941828,  29.258557,   4.355483, -12.789007],
            [ 17.144441,  38.735306,   5.310281, -11.783760],
            [ 18.149161,  46.697123,   6.117972, -10.988521],
            [ 20.612840,  51.503650,   6.553066, -10.411781],
            [ 23.895109,  52.225773,   6.406608,  -9.888725],
            [ 27.258984,  49.052029,   5.525431,  -9.168113],
            [ 29.932976,  43.174556,   3.836086,  -7.980032],
            [ 31.222610,  36.392305,   1.353120,  -6.140754],
            [ 30.848310,  30.527326,  -1.834041,  -3.714370],
            [ 29.207536,  26.757369,  -5.564892,  -1.074081],
            [ 27.219839,  25.284313,  -9.585259,   1.264020],
            [ 25.860539,  25.618417, -13.551476,   2.952189],
            [ 25.792902,  27.124858, -17.098890,   4.026939],
            [ 27.266851,  29.301111, -19.896709,   4.875640],
            [ 30.161152,  31.735027, -21.626337,   5.983374],
            [ 34.123444,  34.065150, -21.975685,   7.600887],
            [ 38.788384,  36.068610, -20.726788,   9.534640],
            [ 43.912527,  37.677380, -17.856114,  11.180246],
            [ 49.279619,  38.851534, -13.565786,  11.735397],
            [ 54.466105,  39.495382,  -8.266175,  10.480137],
            [ 58.744964,  39.498865,  -2.538928,   7.065214],
            [ 61.360522,  38.812140,   2.939070,   1.692492],
            [ 62.040778,  37.530663,   7.521833,  -4.898670],
            [ 61.138180,  35.989429,  10.745451, -11.583101],
            [ 59.249341,  34.709463,  12.461469, -17.144454],
            [ 56.954627,  34.151805,  12.879519, -20.719478],
            [ 54.815602,  34.491895,  12.468383, -22.138523],
            [ 53.244232,  35.563882,  11.759196, -21.891463],
            [ 52.324279,  36.992566,  11.156407, -20.800865],
            [ 51.884391,  38.413497,  10.841960, -19.651835],
            [ 51.742262,  39.594722,  10.793087, -18.916917],
            [ 51.786683,  40.451519,  10.880703, -18.674119],
            [ 51.897246,  41.012688,  10.981452, -18.743869],
            [ 51.971293,  41.348247,  11.036082, -18.901777],
            [ 51.990808,  41.542466,  11.042002, -19.014432],
            [ 51.954094,  41.688032,  11.016661, -19.058965],
            [ 51.803243,  41.840487,  10.966974, -19.067961],
            [ 51.469690,  42.016321,  10.882518, -19.072310],
            [ 50.974045,  42.215194,  10.749125, -19.079777],
            [ 50.449028,  42.415739,  10.566096, -19.089701],
            [ 50.020388,  42.609922,  10.355386, -19.128340],
            [ 49.713847,  42.825261,  10.153965, -19.246267],
            [ 49.535122,  43.048231,   9.992190, -19.450124],
            [ 49.553901,  43.154800,   9.882318, -19.657175],
            [ 49.852826,  42.969574,   9.825851, -19.743342],
            [ 50.443753,  42.409690,   9.820369, -19.623993],
            [ 51.225746,  41.581394,   9.859702, -19.302721],
            [ 52.003319,  40.728499,   9.934018, -18.880721],
            [ 52.601024,  40.060365,  10.026046, -18.511111],
            [ 53.015659,  39.576901,  10.103698, -18.307800],
            [ 53.573910,  38.853257,  10.110751, -18.214536],
            [ 54.507092,  37.546040,   9.997658, -18.182652],
            [ 55.959110,  35.345033,   9.714812, -18.143533],
            [ 58.156790,  31.844124,   9.207502, -17.992034]])

    
    coupling_dir = os.path.join(os.path.dirname(__file__), "coupling")

    # Load the saved model parameters and reconstruct a proper DMPModel object
    model_npz = np.load(os.path.join(coupling_dir, "dmp_model.npz"))
    cw_npz = np.load(os.path.join(coupling_dir, f"S{subject:02d}_curvature_weights_mean.npz"))

    dmp_model = DMPModel(
        weights=np.asarray(model_npz["weights"], dtype=float),
        centers=np.asarray(model_npz["centers"], dtype=float),
        widths=np.asarray(model_npz["widths"], dtype=float),
        alpha_canonical=float(np.atleast_1d(model_npz["alpha_canonical"])[0]),
        alpha_transformation=float(np.atleast_1d(model_npz["alpha_transformation"])[0]),
        beta_transformation=float(np.atleast_1d(model_npz["beta_transformation"])[0]),
        tau=float(np.atleast_1d(model_npz["tau"])[0]),
        n_joints=int(np.asarray(model_npz["weights"]).shape[0]),
        curvature_weights=np.asarray(cw_npz["curvature_weights"], dtype=float),
    )

    # Roll out with and without curvature weights
    q0 = demo[0]
    g = demo[-1]
    tau = 1.0
    dt = 1.0 / (demo.shape[0] - 1)
    personalized_q = rollout_simple_with_coupling(dmp_model, q0, g, tau, dt)
    personalized_q = np.clip(personalized_q, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])
    return demo, personalized_q


if __name__ == "__main__":

    # Print arrays without scientific notation (no 1.23e+04 formatting)
    np.set_printoptions(suppress=True, floatmode="fixed", precision=4)

    # Get the DMP model and curvature weights for the subject
    q1, personalized_q1 = get_trajectories(subject=1)
    q2, personalized_q2 = get_trajectories(subject=2)
    q3, personalized_q3 = get_trajectories(subject=3)
    q4, personalized_q4 = get_trajectories(subject=4)
    q5, personalized_q5 = get_trajectories(subject=5)
    q6, personalized_q6 = get_trajectories(subject=6)
    q7, personalized_q7 = get_trajectories(subject=7)
    q9, personalized_q9 = get_trajectories(subject=9)
    q10, personalized_q10 = get_trajectories(subject=10)

    # Comma-separated (CSV-like) formatting for nicer copy/paste
    print(np.array2string(personalized_q1, separator=", "))
    
 