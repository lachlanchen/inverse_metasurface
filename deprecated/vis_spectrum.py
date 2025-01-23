import re
import matplotlib.pyplot as plt
import numpy as np

# Path to your file containing the captured terminal output
filename = "spectrum_results.txt"

# We'll store parsed data here:
# data_by_csv = {
#    "partial_crys_data/partial_crys_C0.0.csv": {
#        "lambda": [ ... ],
#        "R":      [ ... ],
#        "T":      [ ... ],
#        "RplusT": [ ... ],
#    },
#    ...
# }
data_by_csv = {}

current_csv = None  # track which CSV we are processing

# Regex to match lines like:
# partial_crys_data/partial_crys_C0.0.csv | Row=1, λ=1.040 µm, freq=0.962, (n=3.589, k=0.061)  => R=0.4514, T=0.3104, R+T=0.7618
pattern_line = re.compile(
    r"^(.*?)\s*\|\s*Row=(\d+),\s*λ=([\d.]+)\s*µm,\s*freq=([\d.]+).*R=([\d.]+),\s*T=([\d.]+),\s*R\+T=([\d.]+)"
)

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        
        # Detect lines indicating a new CSV
        # e.g. "Now processing CSV:  partial_crys_data/partial_crys_C0.0.csv"
        if line.startswith("Now processing CSV:"):
            # Extract the CSV filename after the colon
            # e.g. line = "Now processing CSV:	partial_crys_data/partial_crys_C0.0.csv"
            # we can do a split or regex
            parts = line.split("CSV:")
            if len(parts) == 2:
                csv_path = parts[1].strip()
                current_csv = csv_path
                if current_csv not in data_by_csv:
                    data_by_csv[current_csv] = {
                        "lambda": [],
                        "R": [],
                        "T": [],
                        "RplusT": [],
                    }
            continue
        
        # Otherwise, try matching the main data line
        match = pattern_line.search(line)
        if match:
            csv_match   = match.group(1).strip()  # e.g. partial_crys_data/partial_crys_C0.0.csv
            row_str     = match.group(2)          # "1"
            lam_str     = match.group(3)          # "1.040"
            freq_str    = match.group(4)          # "0.962"
            r_str       = match.group(5)          # "0.4514"
            t_str       = match.group(6)          # "0.3104"
            rt_str      = match.group(7)          # "0.7618"
            
            # Convert to float
            lam_val  = float(lam_str)
            r_val    = float(r_str)
            t_val    = float(t_str)
            rt_val   = float(rt_str)
            
            # Store in our dictionary
            if current_csv not in data_by_csv:
                data_by_csv[current_csv] = {
                    "lambda": [],
                    "R": [],
                    "T": [],
                    "RplusT": [],
                }
            
            data_by_csv[current_csv]["lambda"].append(lam_val)
            data_by_csv[current_csv]["R"].append(r_val)
            data_by_csv[current_csv]["T"].append(t_val)
            data_by_csv[current_csv]["RplusT"].append(rt_val)

# Now we have a dictionary of CSVs -> arrays of lambda, R, T, R+T.
# Let's plot everything in one figure, with subplots for R, T, and R+T.

plt.figure(figsize=(10, 10))

# Subplot for Reflection
plt.subplot(3,1,1)
for csvfile, vals in data_by_csv.items():
    lam_arr = np.array(vals["lambda"])
    r_arr   = np.array(vals["R"])
    
    # Extract something like "C0.4" from the file name for the legend
    # e.g. partial_crys_data/partial_crys_C0.4.csv => "C0.4"
    # Adjust the pattern if your file name differs
    label_str = csvfile
    match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
    if match_c:
        label_str = "C" + match_c.group(1)
    
    plt.plot(lam_arr, r_arr, label=label_str)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection (R)")
plt.title("Reflection vs. Wavelength")
plt.grid(True)
plt.legend()

# Subplot for Transmission
plt.subplot(3,1,2)
for csvfile, vals in data_by_csv.items():
    lam_arr = np.array(vals["lambda"])
    t_arr   = np.array(vals["T"])
    
    label_str = csvfile
    match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
    if match_c:
        label_str = "C" + match_c.group(1)
    
    plt.plot(lam_arr, t_arr, label=label_str)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Transmission (T)")
plt.title("Transmission vs. Wavelength")
plt.grid(True)
plt.legend()

# Subplot for R+T
plt.subplot(3,1,3)
for csvfile, vals in data_by_csv.items():
    lam_arr  = np.array(vals["lambda"])
    rt_arr   = np.array(vals["RplusT"])
    
    label_str = csvfile
    match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
    if match_c:
        label_str = "C" + match_c.group(1)
    
    plt.plot(lam_arr, rt_arr, label=label_str)
plt.xlabel("Wavelength (µm)")
plt.ylabel("R + T")
plt.title("R + T vs. Wavelength")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

