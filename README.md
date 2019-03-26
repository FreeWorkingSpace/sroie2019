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
If the function is not described, means you do not need to change its content, or changing the content will not actually improve the result.
### 1. Data Loading
####Modify code at tb_data.py
Defines the pipeline of loading data and some basic loading methods<br />
```
函数名：fetch_detection_data
可以更改的区域为：
subset = Arbitrary_Dataset(args,... , pre_process=[eatimate_angle], augmentation=[aug_sroie()])
可以更改pre-process和augmentation两项
其中pre-process的候选在tb_preprocess.py中定义, 分别为: 
eatimate_angle: 预测旋转的角度，并进行对比度增强, 慢
estimate_angle_and_crop_area: 预测旋转的角度和可以切除的区域，并进行对比度增强, 极慢
clahe_inv: 仅做对比度增强, 最快
其中augmentation的候选在tb_augment.py中定义, 现在好用的就只有: aug_sroie()
```

### 2. Architecture Modification
####Modify code at tb_model.py
Defines the construction of the model structure and forward graph
```
变量名：cfg

函数名：fetch_detection_data
可以更改的区域为：
subset = Arbitrary_Dataset(args,... , pre_process=[eatimate_angle], augmentation=[aug_sroie()])
可以更改pre-process和augmentation两项
其中pre-process的候选在tb_preprocess.py中定义, 分别为: 
eatimate_angle: 预测旋转的角度，并进行对比度增强, 慢
estimate_angle_and_crop_area: 预测旋转的角度和可以切除的区域，并进行对比度增强, 极慢
clahe_inv: 仅做对比度增强, 最快
其中augmentation的候选在tb_augment.py中定义, 现在好用的就只有: aug_sroie()
```

### 3. Check your Modification
#### 3-1. Check augmentation
#### 3-2. Check prior box(default boxes)



