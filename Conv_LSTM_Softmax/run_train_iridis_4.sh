#!/bin/bash

#!/bin/bash

#PBS -q gpu
#PBS -m ae -M chk1g16@soton.ac.uk
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00


# load environment
cd /home/rg3g15/Speech-Recognition/Conv_LSTM_Softmax
module load conda
source activate py3venv

# run script
python train.py

