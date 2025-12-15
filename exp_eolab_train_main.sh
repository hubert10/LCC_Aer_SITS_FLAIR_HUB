#!/bin/bash 
# #SBATCH --job-name=exp_lcc_flair_hub_eolab
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:a100m40:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G
# #SBATCH --time=48:00:00
# #SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output logs/exp_lcc_flair_hub_eolab_%j.out
# #SBATCH --error logs/exp_lcc_flair_hub_eolab_%j.err

export CONDA_ENVS_PATH=$HOME/.conda/envs
DATA_DIR="/my_data/"
export DATA_DIR
source /home/eouser/flair_venv/bin/activate
which python
cd $HOME/exp_2026/LCC_Aer_SITS_FLAIR_HUB
python main.py --config ./configs/train_main/ 
