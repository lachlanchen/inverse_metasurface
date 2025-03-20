#!/usr/bin/env python3

def main():
    # Set to a single CSV path you want to “test” on the first 11 lines:
    csv_file = "merged_myrun_seed12345_g40/myrun_seed12345_g40_nQ1_00001-10000_chunk1.csv"

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Limit to the first 11 lines (header + next 10)
    lines = lines[:11]

    for i, line in enumerate(lines):
        line = line.rstrip("\n")

        if i == 0:
            # Header line: just print as-is
            print("[HEADER]", line)
        else:
            # Attempt to split off the last column at the final comma
            parted = line.rsplit(",", 1)
            if len(parted) < 2:
                print(f"[WARN] line {i} has no last column => {line}")
                continue

            main_part, last_col = parted[0], parted[1]

            # Wrap the last column in quotes
            new_last_col = f"\"{last_col}\""
            new_line = main_part + "," + new_last_col

            print("\n[DATA] OLD:", line)
            print("[DATA] NEW:", new_line)

    print("\n[DONE] This was a DRY RUN (first 11 lines only).")

if __name__ == "__main__":
    main()

