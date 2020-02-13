#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH -o trainer-stdout.log
#SBATCH -e trainer-stderr.log
#SBATCH --constraint="avx"

module load Python
module load cudnn
source venv/bin/activate
python trainer.py
