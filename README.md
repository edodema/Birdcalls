# Birdcalls

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/code-Lightning-blueviolet"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>
    <a href="https://streamlit.io/"><img alt="UI: streamlit" src="https://img.shields.io/badge/ui-streamlit-orange"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://github.com/lucmos/nn-template"> <img src="https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray" alt=""></a> 
</p>

A bird classification expert system based on the <a href="https://www.kaggle.com/c/birdclef-2021/data"> BirdCLEF2021 </a> task and dataset, in the [report](report/main.pdf) only one model is described even though in the code we have more.
This is due to two methods being developed but only the second one revealed to be computationally feasible.

N.b. I did **not** participate to the challenge nor did submit any model.

## Structure
```
.
├── conf
│   ├── data
│   ├── demo
│   ├── hydra
│   ├── logging
│   ├── model
│   ├── optim
│   └── train
├── models
├── notebooks
├── report
│   └── images
└── src
    ├── common
    ├── demo
    ├── pl_data
    └── pl_modules
```

## Requirements
- Python 3.9
- `kaggle`
- `pip`
## Installation

First download the repository and place yourself in it.
```angular2html
$ git clone https://github.com/edodema/Birdcalls.git
$ cd Birdcalls
```
Then download dependencies, keep in mind that process could require one hour or more, depending on your connection.
If one or more commans fail execute them separately, to get more information on the script run `./setup.sh -h`.
```
$ chmod u+x setup.sh
$ ./setup.sh -dmo
```

**Remember to configure the  `.venv` environment, change absolute paths in `.env.template` and rename it to `.env`!**

## Demo
```
$ chmod u+x ./src/demo/run.sh
$ ./src/demo/run.sh
```
![](report/images/demo.gif)
