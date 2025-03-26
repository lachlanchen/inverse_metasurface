import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ---------------------------
# 1. Load Data from CSV files
# ---------------------------
def load_csv_data(filename):
    """Load data from CSV files, handling potential formatting issues."""
    wavelengths = []
    values = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows
            if not row:
                continue
                
            # Ensure we have at least two columns
            if len(row) < 2:
                continue
                
            try:
                wavelength = float(row[0].strip())
                value = float(row[1].strip())
                
                wavelengths.append(wavelength)
                values.append(value)
            except ValueError:
                # Skip rows that can't be converted to float (likely headers)
                continue
    
    return np.array(wavelengths), np.array(values)

# Load data
try:
    wavelengths_amor_n, n_amor = load_csv_data("nk_crystalline_amorphous/aGSST_n.csv")
    wavelengths_amor_k, k_amor = load_csv_data("nk_crystalline_amorphous/aGSST-k.csv")
    wavelengths_crys_n, n_crys = load_csv_data("nk_crystalline_amorphous/cGSST_n.csv")
    wavelengths_crys_k, k_crys = load_csv_data("nk_crystalline_amorphous/cGSST_k.csv")
    
    print("Data loaded successfully!")
    print(f"Amorphous n: {len(wavelengths_amor_n)} points, range {min(wavelengths_amor_n)}-{max(wavelengths_amor_n)}")
    print(f"Amorphous k: {len(wavelengths_amor_k)} points, range {min(wavelengths_amor_k)}-{max(wavelengths_amor_k)}")
    print(f"Crystalline n: {len(wavelengths_crys_n)} points, range {min(wavelengths_crys_n)}-{max(wavelengths_crys_n)}")
    print(f"Crystalline k: {len(wavelengths_crys_k)} points, range {min(wavelengths_crys_k)}-{max(wavelengths_crys_k)}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure the CSV files are in the correct location.")
    exit(1)

# ---------------------------
# 2. Interpolate data to a common wavelength grid
# ---------------------------
# Create a common wavelength grid from 600 to 1100 nm (100 points)
wavelengths_common = np.linspace(600, 1100, 100)

# Interpolate all data onto this common grid
n_amor_interp = np.interp(wavelengths_common, wavelengths_amor_n, n_amor)
k_amor_interp = np.interp(wavelengths_common, wavelengths_amor_k, k_amor)
n_crys_interp = np.interp(wavelengths_common, wavelengths_crys_n, n_crys)
k_crys_interp = np.interp(wavelengths_common, wavelengths_crys_k, k_crys)

# Build complex permittivities: epsilon = (n + i*k)^2
e_crys = (n_crys_interp + 1j*k_crys_interp)**2
e_amor = (n_amor_interp + 1j*k_amor_interp)**2

# ---------------------------
# 3. Maxwell–Garnett equation
# ---------------------------
def compute_epsilon_eff(e_a, e_c, C):
    """
    Compute effective permittivity using Maxwell-Garnett equation.
    
    e_eff = (1 + 2L)/(1 - L),
    where L = C*( (e_c - 1)/(e_c + 2) ) + (1-C)*( (e_a - 1)/(e_a + 2) ).
    
    Parameters:
    e_a : array_like
        Permittivity of amorphous phase
    e_c : array_like
        Permittivity of crystalline phase
    C : float
        Crystallinity (0 to 1)
        
    Returns:
    array_like
        Effective permittivity
    """
    term_c = (e_c - 1) / (e_c + 2)
    term_a = (e_a - 1) / (e_a + 2)
    L = C*term_c + (1 - C)*term_a
    return (1 + 2*L) / (1 - L)

# ---------------------------
# 4. Crystallinity sweep
# ---------------------------
# Create 11 points from 0.0 to 1.0
crystallinities = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, 0.2, ... 1.0

# Dictionary to hold all results
results = {}

for C in crystallinities:
    e_eff = compute_epsilon_eff(e_amor, e_crys, C)
    sqrt_e = np.sqrt(e_eff)
    n_eff = sqrt_e.real
    k_eff = sqrt_e.imag
    results[C] = {
        'wavelengths': wavelengths_common,
        'n': n_eff,
        'k': k_eff
    }

# ---------------------------
# 5. Make a folder & save CSV
# ---------------------------
output_dir = "gsst_partial_crys_data"
os.makedirs(output_dir, exist_ok=True)

for C in crystallinities:
    lam_arr = results[C]['wavelengths']
    n_arr = results[C]['n']
    k_arr = results[C]['k']
    
    # Construct filename
    fname = f"{output_dir}/gsst_partial_crys_C{C:.1f}.csv"
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Wavelength_nm", "n_eff", "k_eff"])
        for lam, nval, kval in zip(lam_arr, n_arr, k_arr):
            writer.writerow([f"{lam:.1f}", f"{nval:.6f}", f"{kval:.6f}"])
    print(f"Saved: {fname}")

# ---------------------------
# 6. Plot results
# ---------------------------
plt.figure(figsize=(15, 7))

# Use a colormap for better visualization of different crystallinity values
colors = plt.cm.viridis(np.linspace(0, 1, len(crystallinities)))

# (a) n
plt.subplot(1, 2, 1)
for i, C in enumerate(crystallinities):
    plt.plot(
        results[C]['wavelengths'], 
        results[C]['n'], 
        label=f'C={C:.1f}',
        color=colors[i],
        linewidth=2
    )
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Refractive Index (n)', fontsize=12)
plt.title('GSST Partial Crystallization: n', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title="Crystallinity")

# (b) k
plt.subplot(1, 2, 2)
for i, C in enumerate(crystallinities):
    plt.plot(
        results[C]['wavelengths'], 
        results[C]['k'], 
        label=f'C={C:.1f}',
        color=colors[i],
        linewidth=2
    )
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Extinction Coefficient (k)', fontsize=12)
plt.title('GSST Partial Crystallization: k', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title="Crystallinity")

plt.tight_layout()

# Save the figure
plt.savefig("gsst_partial_crystallization.png", dpi=300, bbox_inches='tight')
print("Saved: gsst_partial_crystallization.png")

# Show the figure
plt.show()
