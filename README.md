# The Meta-Representation Hypothesis
Meta-representation is a higher-order representation, akin to the human ability to abstract concepts. 

In this paper, we propose a novel theoretical framework for analyzing the generalization problem of deep reinforcement learning. Based on mild assumptions, we theoretically prove that improving the policy robustness to irrelevant features enhances generalization performance.

# Introduction
![abstraction](results/abstraction.jpg)

_Pablo Picasso’s The Bull. By focusing on and exaggerating specific details, rather than trying to capture every detail realistically, artists can convey the core meaning or essence of the subject more powerfully._

![motivation](results/Plato.jpg)

_Plato's Allegory of the Cave. The allegory describes a group of prisoners who are bound in a cave and can only see the shadows on the wall. They have developed their own understanding of the names and characteristics of the shadows, but this understanding is based on a fake perception of the real world._

# Motivation
![motivation](results/motivation.png)

_We propose a hypothesis that deep mutual learning (DML) technique can help agents learn meta-representations of high-dimensional observations, thus improving generalization performance._

# Results

## Training performance
![train](results/all_train_return.png)

## Generalization performance
![test](results/all_test_return.png)

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
