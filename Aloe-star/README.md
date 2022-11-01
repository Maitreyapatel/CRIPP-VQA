
<div  align="center">

# Aloe*+BERT
<a  href="https://pytorch.org/get-started/locally/"><img  alt="PyTorch"  src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a  href="https://pytorchlightning.ai/"><img  alt="Lightning"  src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a  href="https://hydra.cc/"><img  alt="Config: Hydra"  src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a  href="https://github.com/ashleve/lightning-hydra-template"><img  alt="Template"  src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)

[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

  

## Description

  

Official implementation of the Aloe*+BERT for CRIPP-VQA dataset. 
Note: Hyper-parameters might differ from the mentioned in the paper.

  

## How to run
### Install dependencies
```bash
# clone project
git@github.com:Maitreyapatel/CRIPP-VQA.git
cd CRIPP-VQA/Aloe-star

# [OPTIONAL] create conda environment
conda env create -f environment.yml
conda activate aloe_star
```

### Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```bash
# train on CPU
python train.py trainer.gpus=0 experiment=<experiment-name>

# train on 1 GPU
python train.py trainer.gpus=1 experiment=<experiment-name>

# train on >1 GPUs
python train.py trainer=ddp trainer.gpus=2 experiment=<experiment-name>
```
Here,  `<experiment-name>` can have four different possible training settings: `combine`, `descriptive`, `counterfactual`, `planning` . Current version of the `combine` experiment only trains the model on descriptive and counterfactual questions only.

#### Training with pre-trained Aloe*+BERT base model
It is worth noting that Aloe*+BERT model should not be trained on counterfactual questions from scratch. Because in our experiments we observe that until the performance of Aloe improves on descriptive questions, it will not improve on counterfactual/planning tasks at all. 

Feel free to train the `combine` experiment from scratch. But be ready to wait for hours until model starts learning the counterfactual questions.

The effective way of training the Aloe*+BERT model is to first train the model for several epochs on only `descriptive` QAs and then use this pre-trained weight to initialize the `combine` experiment. Follow the below steps:

```bash
# This process will take more than a day to finish (depending upon the avaialbe resources) 
python train.py trainer=ddp trainer.gpus=2 experiment=descriptive trainer.max_epochs=25

# This process will take more another two days to finish (depending upon the avaialbe resources)
## ckpt can be found inside logs/experiments
python train.py trainer=ddp trainer.gpus=2 experiment=combine trainer.max_epochs=25 model.descriptive_ckpt_path=</absolute/path/to/ckpt/file>

# Fine-tune for planning-only task (this step is pretty quick)
python train.py trainer=ddp trainer.gpus=2 experiment=planning trainer.max_epochs=50 model.descriptive_ckpt_path=</absolute/path/to/ckpt/file>

# [OPTIONAL] [SUGGESTED] further finetune the model on only counterfactual tasks (this step is pretty quick)
python train.py trainer=ddp trainer.gpus=2 experiment=counterfactual trainer.max_epochs=50 model.descriptive_ckpt_path=</absolute/path/to/ckpt/file>
```

#### You can override any parameter from command line like this
```bash
python train.py experiment=<experiment-name> trainer.max_epochs=20 datamodule.batch_size=64
```

#### About logging
This default assumes wandb as default logger for the experiments. If you don't want to use the wandb as logger then you can use following command to run the experiments with PyTorch-Lightning default logger.
```bash
python train.py experiment=<experiment-name> logger=default callbacks=default
```

### Evaluations
Note: The current version do not support the CRIPP-VQA leaderboard specific evaluation. Please wait until we release the evaluation as a part of the Aloe*+BERT pipeline. 
In a meantime, please follow the instructions to follow the general evaluations.

## Dataset preparation
* Download the datasets and features from [link](https://maitreyapatel.com/CRIPP-VQA/#dataset).
* Extract and store them in `data/` folder.

### Structure of the `data` folder:
```
data/
├── descriptive_qa.json
├── counterfactual_qa.json
├── planning_qa.json
├── predefined_objects.csv
```

## Issues
Regarding any technical issues, feel free to raise the issue. 
For other doubts or concerns, feel free to reach out to the authors.

## Pipelined releases

 - [x] ~~Release initial version of the reproducible Aloe*+BERT pipeline.~~
 - [ ] Release the combined training pipeline on all tasks.
 - [ ] [additional feature] support for different SSL methods to pre-train Aloe*.
 - [ ] Release the Aloe*+BERT model specific evaluation.


## Issues
For technical concerns please create the GitHub issues. A quick way to resolve any issues would be to reach out to the author at [maitreya.patel@asu.edu](mailto:maitreya.patel@asu.edu).