# SROIE 2019


## Installation
Clone this repo under ~/Documents:
```
git clone --recurse-submodules https://github.com/loveorchids/sroie2019 ~/Documents/sroie2019
```

## Requirement
Python:  3.5.2 or higher
```
pip install -r requirements.txt
```


## Prepare Data
1. Download dataset from SROIE 2019 to PATH_TO_DATASET
2. Create path ~/Pictures/dataset/ocr/ and move SROIE dataset under it
```
cd ~/Pictures
mkdir dataset
cd dataset
mkdir ocr
cd PATH_TO_DATASET
# rename dataset
mv PATH_TO_DATASET/DATASET_NAME PATH_TO_DATASET/SROIE2019
# move dataset
mv -r PATH_TO_DATASET/SROIE2019 ~/Pictures/dataset/ocr/
```

### Training
```
# Run in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 textbox.py
```

### Testing
```
# Run in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 tb_test.py (some args)
```

## Pretrained Model for Task1
The F1-score on SROIE 2019 test set is expected to be 94%.
```
https://drive.google.com/open?id=1uZnBuLm_DLKwNukYhwbrbDxUmLa2XtPe
```
