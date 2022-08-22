# OCTRetImageGen_CLcVAE
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

*Contrastive Learning for Generating Optical Coherence Tomography Images of the Retina

## About this Repository
---
The repository is intended to provide supplementary materials and source code of the experimentes conducted in the paper. The paper is submitted SASHIMI2022 Workshop as part of MICCAI 2022 Conference. 

<strong>The supplementary material</strong> can be found under `reports` folder. To acces it,click [here](/reports/Supplementary_material.pdf).
## Repo Structure
---
Inspired by [Cookie Cutter Data Science](https://github.com/drivendata/cookiecutter-data-science)

```
├── LICENSE
├── README.md                   <- The top-level README for users of this project.
├── INSTALLATION.md             <- Guidelines for users on how to install libraries/tools to conduct experiments.
├── DATAONBOARDING.md           <- Information on how to download and use the data.
├── EXPERIMENTS.md              <- Guidelines for users to conduct experiment reported in the paper.
├── Docker                      <- Dockerfile and bash scripts for building and running docker image for experiments.
├── data
│   ├── oct_test_all.csv        <- The list of images in the test set with path info in csv format.
│   └── oct_train_all.csv       <- The list of images in train set with path info in csv format.
│   └── oct_train_filtered.csv  <- The list of images in filtered train set with path info in csv format.
│
├── models                      <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks                   <- Jupyter notebooks for exploratory data analysis and model training.
│
├── reports                     <- Generated supplementary materials.
│
│
├── src                         <- Source code for use in this project.
│   │
│   ├── dataloader              <- Scripts to generate data for training.
│   │
│   ├── models                  <- Scripts to build model for training.
│   │
│   ├── utils                   <- Scripts for utility functions.
│   │
│   └── train_config.py         <- Training configurations.
│   
└── start_jupyter_notebook.sh   <- Start jupyter notebook. 
└── download_data.sh            <- Download and extract OCT2017 data set.
└── requirements.txt            <- Required python libs to install. 
```
## Getting Started
---
Please follow the insturction here to install development stack, dowload the data and conduct experiments. 

* [INSTALL.md](INSTALL.md): Follow the instructions in this file to install development stack.
* [DATAONBOARDING.md](DATAONBOARDING.md): Follow the instructions in this file to download the data
* [EXPERIMENTS.md](EXPERIMENTS.md): Follow the instructions in this file to conduct Exploratory data analysis(EDA) and train models.


## Supplementary Materials

We also provide the supplementary figures and algorithms mentioned in the repo. One may find more information regarding them under [reports](#reports) directory.

**Maintainers**
---
[Sinan Kaplan](https://www.linkedin.com/in/kaplansinan/)

🎯 Roadmap
---

### Checklist for setting an online repository 

- [x] Add a README file
- [x] Add a [CONTRIBUTING](CONTRIBUTING.md) file
- [x] Add a [LICENSE](LICENSE.md)
