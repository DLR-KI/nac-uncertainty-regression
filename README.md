<!--
SPDX-FileCopyrightText: 2026 DLR e.V.

SPDX-License-Identifier: MIT
-->

# Revisiting Neuron Activation Coverage for Uncertainty Estimation

[![Language](https://img.shields.io/github/languages/top/DLR-KI/nac-uncertainty-regression?style=flat)](https://github.com/DLR-KI/nac-uncertainty-regression)

[![GitHub Stars](https://img.shields.io/github/stars/DLR-KI/nac-uncertainty-regression.svg?style=social&label=Star)](https://github.com/DLR-KI/nac-uncertainty-regression) 
&nbsp;
[![GitHub Downloads](https://img.shields.io/github/downloads/DLR-KI/nac-uncertainty-regression/total?style=social)](https://github.com/DLR-KI/nac-uncertainty-regression/releases) 

![License](https://img.shields.io/github/license/DLR-KI/nac-uncertainty-regression)
&nbsp;
![Size](https://img.shields.io/github/repo-size/DLR-KI/nac-uncertainty-regression?style=flat)
&nbsp;
[![Issues](https://img.shields.io/github/issues/DLR-KI/nac-uncertainty-regression?style=flat)](https://github.com/DLR-KI/nac-uncertainty-regression/issues)
&nbsp;
[![Latest Release](https://img.shields.io/github/v/release/DLR-KI/nac-uncertainty-regression?style=flat)](https://github.com/DLR-KI/nac-uncertainty-regression/)
&nbsp;
[![Release Date](https://img.shields.io/github/release-date/DLR-KI/nac-uncertainty-regression?style=flat)](https://github.com/DLR-KI/nac-uncertainty-regression/releases)
&nbsp;
[![DOI](https://zenodo.org/badge/1139783478.svg)](https://doi.org/10.5281/zenodo.18479021)

***


## Overview

Code for the paper "**Franke et al.: Revisiting Neural Activation Coverage for Uncertainty Estimation**", accepted for a poster session @ **ESANN 2026**.

Contains minimal torch reimplementation of <https://github.com/BierOne/ood_coverage>, extended by a novel formulation for regression problems.
Only the uncertainty estimation function is re-implemented.

## Install

We recommend using [UV](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.
Run ```uv sync``` to create the _.venv_.
Then run ```uv run <file>``` or activate the .venv by using ```./.venv/bin/activate```.
To run a script use ```python <file>```, like normal.

If not using _UV_, install _python3.13_ and create the _.venv_ derived from the [pyproject.toml](./pyproject.toml)  in the repository root.

For using only the NAC-Wrapper in your own project, we recommend doing ```uv add git+https://github.com/DLR-KI/nac-uncertainty-regression```.
Alternatively you can simply copy ```nac_uncertainty_regression/nac.py``` into your project.

## Usage (Minimum Runnable Example)

```python
import torch
from nac_uncertainty_regression import NACWrapper
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import Imagenette, MNIST
from torchvision.transforms.v2 import Compose, RGB

# init pretrained model + wrap it
model = resnet18(pretrained=True)
model = NACWrapper(model, 
                   layer_name_list=[
                       "layer1.1.bn2",      # use dot notation to access nested layers
                       "layer2.1.bn2",
                       "layer3.1.bn2",
                       "layer4.1.bn2",
                       "fc"
                   ])

transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

# define your data somehow - insert your own data here, just make sure the id_data is actually part of your model's training distribution
id_data_loader_fit = torch.utils.data.DataLoader(torch.utils.data.Subset(Imagenette(root="data/imagenette", download=True, transform=transform), indices=range(1024)), batch_size=32)
id_data_loader_eval = torch.utils.data.DataLoader(Imagenette(root="data/imagenette", download=True, transform=transform, split="val"), batch_size=32)
ood_data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(MNIST(root="data/mnist", download=True, transform=Compose([RGB(), transform])), indices=range(32)), batch_size=32)

# IMPORTANT: Do NOT use 'with torch.no_grad()', NACWrapper needs gradients internally! 
# It will check if gradients are enabled and throw an error if not.

# do some forward passes with I.D. Data to init the wrapper distribution
model.train()
for x, y in id_data_loader_fit:
  # only the call to forward is important 
  _ = model(x)

# set the model to eval to get uncertainty scores
model.eval()
# first get ID uncertainty scores
for x, y in id_data_loader_eval:
    uncertainties = model(x)["uncertainty"]
    mean_uncertainty_id = uncertainties.mean()    # the model outputs are saved in key "out", uncertainty in key "uncertainty"
    std_uncertainty_id = uncertainties.std()
    break

# now get OOD uncertainty scores
for x, y in ood_data_loader:
    uncertainties = model(x)["uncertainty"]
    mean_uncertainty_ood = uncertainties.mean() 
    std_uncertainty_ood = uncertainties.std()
    break

print(f"Mean ID Uncertainty Score: {mean_uncertainty_id}+-{std_uncertainty_id}") 
print(f"Mean OoD Uncertainty Score: {mean_uncertainty_ood}+-{std_uncertainty_ood}") 
```

## Reproducing our Results

Run [experiment_ood.bash](./experiment_ood.bash) to reproduce Figure 1 and [experiment_mse.bash](./experiment_mse.bash) to reproduce Figure 2.
Generate the Figures with [viz.ipynb](./viz.ipynb).

## Important Files

- [nac.py](./nac_uncertainty_regression/nac.py) -> The entire implementation alongside documentation
- [nac_test.py](./nac_test.py) -> Unit Tests

## How to cite our work?

If you find the code or results useful, please cite the paper:

```bibtex
@inproceedings{franke2026revisiting,
  title={Revisiting Neural Activation Coverage for Uncertainty Estimation},
  author={Franke, Benedikt and Förster, Nils and and Köster, Frank and Fischer, Asja and Lange, Markus and Raulf, Arne Peter},
  booktitle={34th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2026 (Scopus; ISSN:)},
  year={2026}
}
```
