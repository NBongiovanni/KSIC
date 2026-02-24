#!/bin/bash
#SBATCH --job-name=KSIC
#SBATCH --output=KSIC-%j.stdout
#SBATCH --error=KSIC-%j.stderr

#SBATCH --mem=44G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=slice
#SBATCH --time=10:00:00

eval "$(conda shell.bash hook)"

conda activate KSIC_v3
which python
python -c "import torch; print(torch.__version__)"

# ---- choix du script Python (vision vs sensor) ----
if [[ "$*" == *"--mode sensor"* ]]; then
  ENTRYPOINT="$HOME/KSIC_v6/entrypoints/run_train_sensor.py"
else
  ENTRYPOINT="$HOME/KSIC_v6/entrypoints/run_train_vision.py"
fi

echo "Entry point: ${ENTRYPOINT}"
echo "Arguments: $@"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

srun python ${ENTRYPOINT} "$@"