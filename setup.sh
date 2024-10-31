
conda create -n crsnet python=3.9 -y

# Activate environment
source activate crsnet || conda activate crsnet

# Update pip
pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt