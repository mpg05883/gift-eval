#!/usr/bin/env bash
set -e  

#  Create and activate conda environment 
conda create --yes --name gift python=3.11.11

# IMPORTANT: conda activate only works if the script runs under a shell
# where conda is initialized, so source the conda setup.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gift

# Install packages in editable mode 
pip install -e .[baseline]
pip install jupyter

#  Create Jupyter kernel to run TS Orchestra notebook
python -m ipykernel install --user --name gift --display-name "gift"

#  Install ts-orchestra in editable mode 
git clone https://github.com/mpg05883/Private-TS-Orchestra.git ts-orchestra
cd ./ts-orchestra
pip install -e .

# Install additional Python packages 
pip install dotted_dict tabulate timecopilot

echo "---------------------------"
echo "Environment setup complete!"
echo "Conda env: gift"
echo "Jupyter kernel: gift"
echo "---------------------------"
