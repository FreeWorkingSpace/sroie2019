# sroie2019

## Installation
Clone this repo under ~/Documents:
```
git clone --recurse-submodules https://github.com/loveorchids/sroie2019 ~/Documents/sroie2019
```

## Prepare Data
1. Download dataset from SROIE 2019
2. Create path ~/Pictures/dataset/ocr/ and move SROIE dataset under it
```
cd ~/Pictures
mkdir dataset
cd dataset
mkdir ocr
mv -r SROIE_2019 ~/Pictures/dataset/ocr/
```

### Run the code
Recommand to use PyCharm to do the job also
```

```

## Hyperparamater Tuning
### 1. Data Loading
Modify code under:
* tb_data.py
> Defines the pipeline of loading data and some basic loading methods
*  tb_pre_processing.py
> Defines all the pre-process functions
* tb_augment.py
> Defines how to augment the images


### 2. Architecture Modification
Modify code under tb_model.py

### 3. Check your Modification
#### 3-1. Check augmentation
#### 3-2. Check prior box(default boxes)



