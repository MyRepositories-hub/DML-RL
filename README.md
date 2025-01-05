# The Meta-Representation Hypothesis

## Our project website is coming soon

## Results

### Training performance

### Generalization performance

## Installation
Create Anaconda environment
```bash
conda create -n procgen_py310 python=3.10 --yes
conda activate procgen_py310
```

Install the requirements
```bash
pip install -r requirements.txt
```

Choose the CUDA version on the official PyTorch website: [https://pytorch.org/](https://pytorch.org/)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Train PPO
```bash
python ppo.py
```

Train PPO with DML
```bash
python main.py
```
