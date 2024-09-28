#!/usr/bin/env python3
#!/bin/bash
# File: sleep_main.sh

# Grid Engine options
#$ -cwd                           # Use current working directory
#$ -j y                           # Merge standard error with standard output
#$ -S /bin/bash                   # Use bash shell
#$ -q all.q                       # Specify queue name
# -l mem_free=100G,scr_free=80G  # Request memory and scratch space
#$ -pe smp 100                    # Parallel environment: 100 slots
#$ -m be                          # Send email at the beginning and end of the job

# Load the necessary environment modules (if any)

# Run the Python script
python3.9 ./sleep_main.py
