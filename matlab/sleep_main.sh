#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
# -l mem_free=100G,scr_free=80G
#$ -pe smp 100 
#$ -m beas
#$ -M ap23710@essex.ac.uk

matlab -nosplash -nodesktop -r "$1; exit"
