#!/bin/bash
set -ex

for i in {1..40}; do
    sbatch run_workers.sh;
done

sbatch run_trainer.sh
