# Distill Donut

## Installation
1. Clone repository
```
git clone https://github.com/hannesSchwanzer/donut_funsd.git
```
2. Install python
3. Install [pytorch](https://pytorch.org/get-started/locally/) (you might need to install cuda first, if you want to use gpu)
4. Install project with requirements
```
pip install .
```

## Download DocVQA
1. Download task 1 from https://rrc.cvc.uab.es/?ch=17&com=downloads (Login is needed; only need to download annotations and images)
2. Create directory docvqa/ at project root with subdirectories documents/ and queries/
3. Extract images in documents/ and annotation files in queries/

## Usage
1. Preprocess the dataset
``` 
python donut_distill/data/preprocess_donut.py
```
2. Finetune / Distill the model. Use --config to overwrite default config. Default configs given in directory configs/. Most parameters describted in donut_distill/config/config.py
```
python donut_distill/training/train.py --config <path_to_config>
```
3. Evaluate:
```
python donut_distill/evaluation/evaluate.py --config <path_to_config>
```


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
