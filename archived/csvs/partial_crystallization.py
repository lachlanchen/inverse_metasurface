import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Define Data
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
n_crys = np.array([
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

# For amorphous
wavelengths_amor = np.array([
    1.039549192, 1.075165591, 1.110781991, 1.14639839,  1.182014789, 1.217631188,
    1.253247588, 1.288863987, 1.324480386, 1.360096785, 1.395713184, 1.431329584,
    1.466945983, 1.502562382, 1.538178781, 1.573795181, 1.60941158,  1.645027979,
    1.680644378, 1.716260778, 1.751877177, 1.787493576, 1.823109975, 1.858726375,
    1.894342774, 1.929959173, 1.965575572, 2.001191972, 2.036808371, 2.07242477,
    2.108041169, 2.143657569, 2.179273968, 2.214890367, 2.250506766, 2.286123166,
    2.321739565, 2.357355964, 2.392972363, 2.428588763, 2.464205162, 2.502142954
])
n_amor = np.array([
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
# 2. Build consistent arrays
#    and compute permittivities
# ---------------------------
# We need to match the same wavelength range for both n & k in each phase
# so let's do an interpolation for k_crys to match the 'wavelengths_crys' array,
# and similarly for k_amor to match 'wavelengths_amor'.

def interpolate_k(wavelengths_kdata, kdata, wavelengths_target):
    """Interpolates k-data onto the target wavelength array."""
    return np.interp(wavelengths_target, wavelengths_kdata, kdata)

k_crys_interp = interpolate_k(k_crys_wavelengths, k_crys, wavelengths_crys)
k_amor_interp = interpolate_k(k_amor_wavelengths, k_amor, wavelengths_amor)

# Convert to complex permittivities:
# epsilon = (n + i*k)^2 = (n^2 - k^2) + 2i n k
# We'll store them as numpy arrays of dtype complex.
e_crys = (n_crys + 1j*k_crys_interp)**2
e_amor = (n_amor + 1j*k_amor_interp)**2

# We'll assume we want the final results on a single "unified" wavelength array.
# For demonstration, let's just do them separately:
#  - For crystalline data, we use wavelengths_crys
#  - For amorphous data, we use wavelengths_amor
#
# Or we could do a single array of all unique wavelengths from both sets.
# For clarity, let's do them individually.

# ---------------------------
# 3. Effective Medium Theory
# ---------------------------
# The EMT equation (Maxwell Garnett form) is:
#   (epsilon_eff - 1)/(epsilon_eff + 2) =
#        C * (epsilon_c - 1)/(epsilon_c + 2) +
#        (1-C)* (epsilon_a - 1)/(epsilon_a + 2)
#
# Solve for epsilon_eff:
#   let L = RHS
#   (epsilon_eff - 1) = L * (epsilon_eff + 2)
#   epsilon_eff - 1 = L * epsilon_eff + 2L
#   epsilon_eff - L*epsilon_eff = 1 + 2L
#   epsilon_eff(1 - L) = 1 + 2L
#   epsilon_eff = (1 + 2L) / (1 - L)

def compute_epsilon_eff(e_a, e_c, C):
    """
    Given arrays of e_a and e_c (same shape),
    returns the effective epsilon array for crystallinity C.
    """
    # (epsilon_c - 1)/(epsilon_c + 2)
    term_c = (e_c - 1.0) / (e_c + 2.0)
    # (epsilon_a - 1)/(epsilon_a + 2)
    term_a = (e_a - 1.0) / (e_a + 2.0)

    # L = C * term_c + (1-C) * term_a
    L = C * term_c + (1 - C) * term_a

    # e_eff = (1 + 2L)/(1 - L)
    # Watch out for potential division by zero if L = 1, but that shouldn't
    # happen with typical permittivity values.
    e_eff = (1.0 + 2.0 * L) / (1.0 - L)
    return e_eff

# ---------------------------
# 4. Crystallinity sweep + Compute n,k from sqrt(e_eff)
# ---------------------------
# Let's pick a few crystallinity values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
# We'll store the results in a dictionary for plotting.
crystallinities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# For demonstration, we'll do the effective medium on the
# intersection of the crystalline & amorphous wavelength arrays if you want
# them on a single plot. Otherwise, we can keep them separate.
# Let's do a single array that merges them. We'll just take the union
# and sort them.

all_wavelengths = np.unique( np.concatenate([wavelengths_crys, wavelengths_amor]) )
# Interpolate e_crys, e_amor onto this merged wavelength grid:
e_crys_interp = np.interp(all_wavelengths, wavelengths_crys, e_crys.real) + 1j*np.interp(all_wavelengths, wavelengths_crys, e_crys.imag)
e_amor_interp = np.interp(all_wavelengths, wavelengths_amor, e_amor.real) + 1j*np.interp(all_wavelengths, wavelengths_amor, e_amor.imag)

results = {}
for C in crystallinities:
    e_eff = compute_epsilon_eff(e_amor_interp, e_crys_interp, C)
    # n_eff, k_eff from sqrt(e_eff)
    # We use np.sqrt for complex sqrt. The real part is n, the imag part is k.
    # Note: np.sqrt chooses the principal branch of the square root in the complex plane.
    # If you'd like continuity, you might want to handle the sign. But for typical optical data,
    # the principal branch is fine.
    sqrt_e = np.sqrt(e_eff)
    n_eff = sqrt_e.real
    k_eff = sqrt_e.imag
    results[C] = {
        'wavelengths': all_wavelengths,
        'n': n_eff,
        'k': k_eff
    }

# ---------------------------
# 5. Visualization
# ---------------------------
# We will produce Figure 2:
#   (a) Refractive index vs. wavelength  [n_eff(lambda, C)]
#   (b) Extinction coefficient vs. wavelength [k_eff(lambda, C)]
plt.figure(figsize=(12,5))

# (a) Refractive Index
plt.subplot(1,2,1)
for C in crystallinities:
    plt.plot(results[C]['wavelengths'], results[C]['n'], label=f'C={C}')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Refractive Index n')
plt.title('(a) Partial Crystallization: n')
plt.grid(True)
plt.legend()

# (b) Extinction Coefficient
plt.subplot(1,2,2)
for C in crystallinities:
    plt.plot(results[C]['wavelengths'], results[C]['k'], label=f'C={C}')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Extinction Coefficient k')
plt.title('(b) Partial Crystallization: k')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


