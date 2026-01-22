<!--
SPDX-FileCopyrightText: 2026 DLR e.V.

SPDX-License-Identifier: MIT
-->

# Revisiting Neuron Activation Coverage for Uncertainty Estimation
## Overview
Code for the paper "Franke et al.: Revisiting Neural Activation Coverage for Uncertainty Estimation", accepted for a poster session @ ESANN 2026.

Contains minimal torch reimplementation of https://github.com/BierOne/ood_coverage, extended by a novel formulation for regression problems. Only the uncertainty estimation function is re-implemented.

## Install
Recommended is [UV](https://docs.astral.sh/uv/getting-started/installation/). 
Run ```uv sync``` to create the .venv.
Then run with ```uv run <file>``` or activate the .venv ```./.venv/bin/activate``` and run as normal ```python <file>```

If not using UV, install python3.13 and create a venv from pyproject.toml

## Usage
```python
from nac import NACWrapper

# init rtained model + wrap it
model = ResNet50(pretrained=True)
model = NACWrapper(model)
# IMPORTANT: Do NOT use 'with torch.no_grad()', NACWrapper needs gradients internally! 
# It will check if gradients are enabled and throw an error if not.

# do some forward passes with I.D. Data to init the wrapper distribution
model.train()
for x, y in id_data_loader:
  # only the call to forward is important 
  _ = model(x)

# set the model to eval to get uncertainty scores
model.eval()
res = model(potentially_ood_batch)
print(res["uncertainty"]) # the model outputs are saved in key "out"

```

## Reproducing our Results
Run [experiment_ood.bash](./experiment_ood.bash) to reproduce figure 1 and [experiment_mse.bash](./experiment_mse.bash) to reproduce figure 2.
Generate the figures with [viz.ipynb](./viz.ipynb).

## Important Files
- [nac.py](./nac.py). -> The entire implementation with doc
- [nac_test.py](./nac_test.py) -> Unit Tests

## How to cite our work?
If you find the code or results useful, please cite the paper
```(bibtex)
@inproceedings{franke2026revisiting,
  title={Revisiting Neural Activation Coverage for Uncertainty Estimation},
  author={Franke, Benedikt and Förster, Nils and and Köster, Frank and Fischer, Asja and Lange, Markus and Raulf, Arne Peter},
  booktitle={34th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2026 (Scopus; ISSN:)},
  year={2026}
}
```