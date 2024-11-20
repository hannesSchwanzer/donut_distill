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
4. Install requirements
```
pip install -r requirements.txt
```

## Usage
1. Preprocess the dataset
``` 
python preprocess_donut.py
```
2. Finetune the model
```
python train.py
```

## ToDos
1. Preprocess data
- [x] Link items from json file
- [x] Convert to Donut format
2. Create Dataset
- [x] Use Donut dataset
3. Train Model
- [x] Setup and load pre-trained Donut model
- [x] Create trainingsloop
- [x] Calculate stats (loss, acc)
- [x] Log on wandb
- [ ] Run training
