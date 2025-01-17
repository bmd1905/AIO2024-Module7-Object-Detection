## Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Use conda
conda create -n object-detection python=3.11 -y
conda activate object-detection
conda install ffmpeg -c conda-forge -y
pip install -r requirements.txt

# Use venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
