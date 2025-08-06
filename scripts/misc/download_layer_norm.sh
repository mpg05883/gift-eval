#SBATCH --job-name=download_layer_norm
#SBATCH --partition=gpuA40x4  
#SBATCH --mem=64GB  
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/%x/out/%A.out
#SBATCH --error=output/logs/%x/err/%A.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Fail on error
set -e

# Load required CUDA module
module load cuda/12.6

# Activate conda
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate gift

# Move into the extension directory
cd /projects/beei/mgee2/YingLong/flash-attention/csrc/layer_norm

# Install ninja if not already
pip install ninja --quiet

# Optional: clean previous builds
python setup.py clean --all || true
rm -rf build/ *.egg-info

# Restrict to A40 architecture (compute capability 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"

# Install the extension
pip install . --use-pep517 --no-build-isolation -v