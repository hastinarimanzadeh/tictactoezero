#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=2GB
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --constraint="avx"

module load Python
module load cudnn
source venv/bin/activate
python worker.py
