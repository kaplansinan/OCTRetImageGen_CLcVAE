# OCTRetImageGen_CLcVAE
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

*Contrastive Learning for Generating Optical Coherence Tomography Images of the Retina

## About this Repository
---
The repository is intended to provide supplementary materials and source code of the experimentes conducted in the paper. The paper is submitted SASHIMI2022 Workshop as part of MICCAI 2022 Conference. 
## Repo Structure
---
Inspired by [Cookie Cutter Data Science](https://github.com/drivendata/cookiecutter-data-science)

```
├── LICENSE
├── README.md          <- The top-level README for users of this project.
├── INSTALLATION.md    <- Guidelines for users on how to install libraries/tools to conduct experiments.
├── DATAONBOARDING.md  <- Information on how to download and use the data.
├── TRAINING.md        <- Guidelines for users to train modeld.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated supplementary material as PDF.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│  └── visualisation  <- Scripts to create exploratory and results oriented visualisations
│       └── visualise.py
└──
```
## Getting Started
---
Please follow the insturction here to install development stack, dowload data and conduct experiments. 

* [INSTALL.md](#INSTALL.md): Follow the insturction in this file to install development stack.
* [DATAONBOARDING.md](#DATAONBOARDING.md): Follow the insturction in this file to download the data
* [EXPERIMENTS.md](#EXPERIMENTS.md): Follow the insturction in this file to conduct explotary data analysis and train models.


## Supplementary Materials

We also provide the supplementary material mentioned in the repo. One may find it under [reports](#reports) directory.

**Maintainers**
---

🎯 Roadmap
---

### Checklist for setting an online repository 

- [ ] Add a README file
- [ ] Add a [CONTRIBUTING](CONTRIBUTING.md) file
- [ ] Add a [LICENSE](LICENSE.md)
