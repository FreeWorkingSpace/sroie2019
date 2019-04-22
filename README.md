# SROIE 2019 Task 1


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
1. Download dataset from SROIE 2019
2. Create path ~/Pictures/dataset/ocr/ and move SROIE dataset under it
```
cd ~/Pictures
mkdir dataset
cd dataset
mkdir ocr
# 移动至数据集文件夹所在位置
cd PATH_TO_DATASET
# 重命名下载后的数据集文件夹
mv PATH_TO_DATASET/DATSET_NAME PATH_TO_DATASET/SROIE2019
# 将数据集移动到指定位置
mv -r PATH_TO_DATASET/SROIE2019 ~/Pictures/dataset/ocr/
```

### Training
```
# Running in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 textbox.py
```


### Testing
```
# Running in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 tb_test.py -mpl tb_003_3.py
```
