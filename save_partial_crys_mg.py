#!/usr/bin/env python3
"""
partial_crys_mg.py

1) We have n,k data for "crys" and "amor".
2) We compute partial crystallization via Maxwell-Garnett effective medium.
3) By default, we create partial_crys_data/partial_crys_CX.csv for X in [0.0..1.0], step=0.1.
4) We allow an optional first command-line argument N (default=100) to decide how many
   uniformly spaced wavelengths we want in the final interpolation.
5) We allow an optional second command-line argument c_value in [0,1]. If provided,
   we only process that specific c, and save it to partial_crys_data_single/partial_crys_C{c}.csv.

Usage examples:
  python partial_crys_mg.py
    -> uses N=100, does c = 0, 0.1, 0.2 ... 1.0
  python partial_crys_mg.py 200
    -> uses N=200, does c = 0, 0.1, 0.2 ... 1.0
  python partial_crys_mg.py 150 0.55
    -> uses N=150, does only c = 0.55, saves to partial_crys_data_single/partial_crys_C0.55.csv
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 0. Parse command-line arguments
# ---------------------------
N = 100           # default number of wavelength points
c_value = None    # if provided, we do single c

if len(sys.argv) > 1:
    # Try to parse the first argument as N
    try:
        N = int(sys.argv[1])
        print(f"Using N = {N} interpolation points.")
    except ValueError:
        print(f"Invalid first argument {sys.argv[1]}, not an integer => using default N=100.")

if len(sys.argv) > 2:
    # Try to parse the second argument as c_value
    try:
        c_val_f = float(sys.argv[2])
        if 0.0 <= c_val_f <= 1.0:
            c_value = c_val_f
            print(f"Detected c_value = {c_value:.3f}")
        else:
            print(f"Warning: c_value = {c_val_f} is out of [0,1]. Ignoring it.")
    except ValueError:
        print(f"Invalid second argument {sys.argv[2]}, not a float => ignoring it.")

# ---------------------------
# 1. Define the data
# ---------------------------
wavelengths_crys = np.array([
    1.039549192, 1.075165591, 1.110781991, 1.14639839, 1.182014789, 1.217631188,
    1.253247588, 1.288863987, 1.324480386, 1.360096785, 1.395713184, 1.431329584,
    1.466945983, 1.502562382, 1.538178781, 1.573795181, 1.60941158, 1.645027979,
    1.680644378, 1.716260778, 1.751877177, 1.787493576, 1.823109975, 1.858726375,
    1.894342774, 1.929959173, 1.965575572, 2.001191972, 2.036808371, 2.07242477,
    2.108041169, 2.143657569, 2.179273968, 2.214890367, 2.250506766, 2.286123166,
    2.321739565, 2.357355964, 2.392972363, 2.428588763, 2.464205162, 2.499821561
])
n_amor = np.array([
    3.588763302, 3.553766089, 3.518768876, 3.490771105, 3.462773335, 3.441775007,
    3.420776679, 3.399778351, 3.385779466, 3.37178058,  3.357781695, 3.350782253,
    3.34378281,  3.329783925, 3.322784482, 3.315785039, 3.308785597, 3.301786154,
    3.294786711, 3.287787269, 3.280787826, 3.280787826, 3.273788384, 3.266788941,
    3.266788941, 3.259789498, 3.259789498, 3.252790056, 3.252790056, 3.252790056,
    3.245790613, 3.245790613, 3.245790613, 3.23879117,  3.23879117,  3.23879117,
    3.231791728, 3.231791728, 3.231791728, 3.231791728, 3.224792285, 3.224792285
])
k_crys_wavelengths = np.array([
    1.03598758,  1.071602851, 1.107218122, 1.142833393, 1.178448663, 1.214063934,
    1.249679205, 1.285294476, 1.320909747, 1.356525017, 1.392140288, 1.427755559,
    1.46337083,  1.4989861,   1.534601371, 1.570216642, 1.605831913, 1.641447184,
    1.677062454, 1.712677725, 1.748292996, 1.783908267, 1.819523537, 1.855138808,
    1.890754079, 1.92636935,  1.961984621, 1.997599891, 2.033215162, 2.068830433,
    2.104445704, 2.140060974, 2.175676245, 2.211291516, 2.246906787, 2.282522058,
    2.318137328, 2.353752599, 2.38936787,  2.424983141, 2.460598411, 2.496213682
])
k_crys = np.array([
    1.110339521, 1.02657927,  0.947007032, 0.871622806, 0.808802618, 0.74598243,
    0.687350254, 0.632906091, 0.582649941, 0.536581803, 0.49888969,  0.461197577,
    0.423505464, 0.394189376, 0.360685276, 0.3355572,   0.314617138, 0.293677075,
    0.272737012, 0.255984962, 0.243420924, 0.230856887, 0.222480862, 0.214104837,
    0.205728812, 0.197352786, 0.188976761, 0.184788749, 0.180600736, 0.172224711,
    0.168036699, 0.163848686, 0.159660674, 0.155472661, 0.151284648, 0.147096636,
    0.142908623, 0.138720611, 0.134532598, 0.134532598, 0.130344586, 0.126156573
])

wavelengths_amor = np.array([
    1.039549192, 1.075165591, 1.110781991, 1.14639839,  1.182014789, 1.217631188,
    1.253247588, 1.288863987, 1.324480386, 1.360096785, 1.395713184, 1.431329584,
    1.466945983, 1.502562382, 1.538178781, 1.573795181, 1.60941158,  1.645027979,
    1.680644378, 1.716260778, 1.751877177, 1.787493576, 1.823109975, 1.858726375,
    1.894342774, 1.929959173, 1.965575572, 2.001191972, 2.036808371, 2.07242477,
    2.108041169, 2.143657569, 2.179273968, 2.214890367, 2.250506766, 2.286123166,
    2.321739565, 2.357355964, 2.392972363, 2.428588763, 2.464205162, 2.502142954
])
n_crys = np.array([
    5.338623961, 5.331624519, 5.324625076, 5.310626191, 5.296627306, 5.28262842,
    5.268629535, 5.247631207, 5.226632879, 5.198635109, 5.177636781, 5.156638453,
    5.135640125, 5.114641797, 5.093643469, 5.065645699, 5.044647371, 5.023649043,
    5.002650715, 4.981652387, 4.960654059, 4.946655174, 4.925656846, 4.911657961,
    4.897659075, 4.88366019,  4.869661305, 4.862661862, 4.848662977, 4.834664092,
    4.834664092, 4.820665206, 4.813665764, 4.806666321, 4.799666878, 4.792667436,
    4.785667993, 4.77866855,  4.771669108, 4.771669108, 4.764669665, 4.763121342
])
k_amor_wavelengths = np.array([
    0.996810782, 1.032426053, 1.068041324, 1.103656595, 1.139271866, 1.174887136,
    1.210502407, 1.246117678, 1.281732949, 1.317348219, 1.35296349,  1.388578761,
    1.424194032, 1.459809303, 1.495424573, 1.531039844, 1.566655115, 1.602270386,
    1.637885656, 1.673500927, 1.709116198, 1.744731469, 1.78034674,  1.81596201,
    1.851577281, 1.887192552, 1.922807823, 1.958423093, 1.994038364, 2.029653635,
    2.065268906, 2.100884177, 2.136499447, 2.172114718, 2.207729989, 2.24334526,
    2.278960531, 2.314575801, 2.350191072, 2.385806343, 2.421421614, 2.457036884
])
k_amor = np.array([
    0.075900423, 0.063336385, 0.050772347, 0.034020297, 0.025644272, 0.017268247,
    0.008892222, 0.008892222, 0.004704209, 0.000516197, 0.000516197, 0.000516197,
    0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197,
    0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197,
    0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197,
    0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197,
    0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197, 0.000516197
])

# ---------------------------
# 2. Interpolate k-data & compute permittivities
# ---------------------------
def interpolate_k(wavelengths_kdata, kdata, wavelengths_target):
    """Interpolates k-data onto the target wavelength array."""
    return np.interp(wavelengths_target, wavelengths_kdata, kdata)

def build_epsilon(n_arr, k_arr):
    """Complex permittivity = (n + i*k)^2."""
    return (n_arr + 1j*k_arr)**2

# Maxwell–Garnett equation
def compute_epsilon_eff(e_a, e_c, C):
    """
    e_eff = (1 + 2L)/(1 - L), 
    where L = C*((e_c - 1)/(e_c + 2)) + (1-C)*((e_a - 1)/(e_a + 2)).
    """
    term_c = (e_c - 1)/(e_c + 2)
    term_a = (e_a - 1)/(e_a + 2)
    L = C*term_c + (1 - C)*term_a
    return (1 + 2*L) / (1 - L)

# ---------------------------
# 3. Build a uniform wavelength grid of size N
# ---------------------------
all_w_min = min(wavelengths_crys.min(), wavelengths_amor.min())
all_w_max = max(wavelengths_crys.max(), wavelengths_amor.max())

all_wavelengths = np.linspace(all_w_min, all_w_max, N)

# Interpolate the crystalline data
k_crys_interp = np.interp(all_wavelengths, k_crys_wavelengths, k_crys)
n_crys_interp = np.interp(all_wavelengths, wavelengths_crys,     n_crys)
e_crys = build_epsilon(n_crys_interp, k_crys_interp)

# Interpolate the amorphous data
k_amor_interp = np.interp(all_wavelengths, k_amor_wavelengths, k_amor)
n_amor_interp = np.interp(all_wavelengths, wavelengths_amor,   n_amor)
e_amor = build_epsilon(n_amor_interp, k_amor_interp)

# ---------------------------
# 4A. If c_value is given, compute only that c and save to a separate folder
# ---------------------------
if c_value is not None:
    single_folder = "partial_crys_data_single"
    os.makedirs(single_folder, exist_ok=True)

    C = c_value
    e_eff = compute_epsilon_eff(e_amor, e_crys, C)
    sqrt_e = np.sqrt(e_eff)
    n_eff  = sqrt_e.real
    k_eff  = sqrt_e.imag

    fname = os.path.join(single_folder, f"partial_crys_C{C:.2f}.csv")
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Wavelength_um", "n_eff", "k_eff"])
        for w, nn, kk in zip(all_wavelengths, n_eff, k_eff):
            writer.writerow([f"{w:.6f}", f"{nn:.6f}", f"{kk:.6f}"])
    print(f"[Single-c mode] Saved c={C:.2f} data to: {fname}")

    # Optionally you can skip the plotting or do a small single plot. 
    # We'll skip the big multi-plot if c_value is provided.
    # sys.exit(0)  # if you want to exit here
else:
    # ---------------------------
    # 4B. Otherwise, do the full sweep in 0.1 steps [0..1]
    # ---------------------------
    sweep_folder = "partial_crys_data"
    os.makedirs(sweep_folder, exist_ok=True)

    crystallinities = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, 0.2, ... 1.0

    for C in crystallinities:
        e_eff = compute_epsilon_eff(e_amor, e_crys, C)
        sqrt_e = np.sqrt(e_eff)
        n_eff  = sqrt_e.real
        k_eff  = sqrt_e.imag

        fname = os.path.join(sweep_folder, f"partial_crys_C{C:.1f}.csv")
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Wavelength_um", "n_eff", "k_eff"])
            for w, nn, kk in zip(all_wavelengths, n_eff, k_eff):
                writer.writerow([f"{w:.6f}", f"{nn:.6f}", f"{kk:.6f}"])
        print(f"Saved: {fname}")

    # ---------------------------
    # 5. (Optional) Plot results (the multi-sweep)
    # ---------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for C in crystallinities:
        cfile = os.path.join(sweep_folder, f"partial_crys_C{C:.1f}.csv")
        data = np.loadtxt(cfile, delimiter=',', skiprows=1)
        wplot, nplot, _ = data.T
        plt.plot(wplot, nplot, label=f'C={C:.1f}')
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Refractive Index n')
    plt.title('(a) Partial Crystallization: n')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for C in crystallinities:
        cfile = os.path.join(sweep_folder, f"partial_crys_C{C:.1f}.csv")
        data = np.loadtxt(cfile, delimiter=',', skiprows=1)
        wplot, _, kplot = data.T
        plt.plot(wplot, kplot, label=f'C={C:.1f}')
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Extinction Coefficient k')
    plt.title('(b) Partial Crystallization: k')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
