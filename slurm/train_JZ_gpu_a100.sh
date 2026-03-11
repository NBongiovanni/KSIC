#!/bin/bash
#SBATCH --job-name=KSIC
#SBATCH --constraint=a100
#SBATCH --account=nxg@a100 # a100 accounting
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --hint=nomultithread

set -e

echo "Running on host: $(hostname)"
nvidia-smi

# ===================== ENV (PATCH ICI) =====================
module purge
module load pytorch-gpu/py3/2.5.0    # <-- si c'est bien le module que tu utilises d'habitude

# Force tes imports à pointer vers ton code (évite les vieux installs)
export PYTHONPATH=$HOME/KSIC/src:$WORK${PYTHONPATH:+:$PYTHONPATH}

# (optionnel mais utile en debug)
echo "PYTHONPATH=$PYTHONPATH"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# ===========================================================

# ---- choix du script Python (vision vs sensor) ----
if [[ "$*" == *"--mode sensor"* ]]; then
  ENTRYPOINT="$HOME/KSIC/entrypoints/run_train_sensor_model.py"
else
  ENTRYPOINT="$HOME/KSIC/entrypoints/run_train_vision_model.py"
fi

echo "Entry point: ${ENTRYPOINT}"
echo "Arguments: $@"

srun python ${ENTRYPOINT} "$@"