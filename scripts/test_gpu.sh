#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney

echo "Hostname: $(hostname)"
echo "GPU info:"
nvidia-smi

python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print("GPU matmul OK, result sum:", y.sum().item())
EOF