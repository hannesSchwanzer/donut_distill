# Donut on FUNSD
Finetuning the Donut model on the FUNSD dataset

## Installation
1. Clone repository
```
git clone https://github.com/hannesSchwanzer/donut_funsd.git
```
2. Get the dataset funsd
```
curl -O https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip
rm dataset.zip
```
3. Install python
4. Install [pytorch](https://pytorch.org/get-started/locally/) (you might need to install cuda first, if you want to use gpu)
5. Install project with requirements
```
pip install .
```

## Usage
1. Preprocess the dataset
``` 
python donut_distill/data/preprocess_donut.py
```
2. Finetune / Distill the model. Use --config to overwrite default config.
```
python donut_distill/training/train.py --config <path_to_config>
```
Example configs in configs/


## Directories and files
- setup.py: Setup file for installing this package and dependencies
- configs/: Directory with config files to overwrite the default training process
- dataset_labeled_human/: Self-annotated dataset of 5 files for funsd
- scripts/: Some helpfull scripts used during project creation, but not necessary for project
- donut_distill/: The main project
    - config/: Config related files
    - data/: Dataset, pre- and postprocessing
    - evaluation/: metrics and evaluation functions
    - models/: student model creation and helper functions
    - training/: Loss, Training and training related functions
