#!/bin/bash


# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=chkaradjov@abv.bg


#SBATCH -p lyceum
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd /lyceum/rt1g16/Speech-Recognition/Conv_LSTM_CTC
module load conda/py3-latest
source activate py3venv

python train.py

