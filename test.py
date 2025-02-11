#!/usr/bin/env python3
import S4

def main():
    # If the S4 extension provides an initialization function, call it.
    if hasattr(S4, 'initialize'):
        S4.initialize()
    print("S4 Python extension compiled and imported successfully!")

if __name__ == "__main__":
    main()

