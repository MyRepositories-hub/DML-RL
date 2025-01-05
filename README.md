# The Meta-Representation Hypothesis
Meta-representation is a higher-order representation, akin to the human ability to abstract concepts.

# Introduction
![abstraction](results/abstraction.jpg)

_Pablo Picasso’s The Bull. By focusing on and exaggerating specific details, rather than trying to capture every detail realistically, artists can convey the core meaning or essence of the subject more powerfully._

# Motivation
![motivation](results/motivation.png)

_We propose a hypothesis that deep mutual learning (DML) technique can help agents learn meta-representations of high-dimensional observations, thus improving generalization performance._

# Installation
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

# Results

## Training performance
![train](results/all_train_return.png)

## Generalization performance
![test](results/all_test_return.png)
