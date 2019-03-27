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

### Run the code
Recommand to use PyCharm
```
# Running in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 textbox.py
```

## Hyperparamater Tuning
If the function is not described, means you do not need to change its content, or changing the content does not have the potential to improve result.

### 1. Modify code for loading data 
#### 1.1 tb_data.py
函数名：**fetch_detection_data**<br>
初始化数据集时可以更改pre-process和augmentation两项<br>
**subset = Arbitrary_Dataset(args,... , pre_process=[eatimate_angle], augmentation=[aug_sroie()])**<br>
* **pre-process**的候选函数在tb_preprocess.py中定义, 分别为: <br>
1 eatimate_angle: 预测旋转的角度，并进行对比度增强, 速度慢<br>
2 estimate_angle_and_crop_area: 预测旋转的角度和可以切除的区域，并进行对比度增强, 速度极慢<br>
3 clahe_inv: 仅做对比度增强, 速度最快<br>

* **augmentation**的候选在tb_augment.py中定义, 现在仅有: aug_sroie() 一项<br>


### 2. Modify code for model architecture
#### 2.1 tb_model.py

变量名：
**cfg**, 更改的方法参照代码中的注释<br>
**SSD的模型的更改**<br>
* 在初始化SSD模型时，有一个默认参数名为connect_loc_to_conf=False，将其设置为True可以使
localization layer的输出与confidence layer相结合，让confidence layer的预测更加准确。
* 在调节完模型的参数后，直接运行tb_model.py来检查是否模型可以正常运行
```
# Running in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 tb_model.py

Output should be similar to: 
CNN output shape: torch.Size([2, 512, 64, 64])
CNN output shape: torch.Size([2, 512, 32, 32])
Loc output shape: torch.Size([2, 56, 64, 64])
Conf output shape: torch.Size([2, 28, 64, 64])
Loc output shape: torch.Size([2, 48, 32, 32])
Conf output shape: torch.Size([2, 24, 32, 32])
torch.Size([2, 69632, 4])
torch.Size([2, 69632, 2])
torch.Size([69632, 4])
```

#### 2.2 tb_utils.py
函数名：**jaccard & intersect**<br>
* 更改他们可以改变计算jaccard distance的方式，目前在匹配的时候比较容易出现横向的错位，也许可以找到更好的计算匹配的模式

### 3. Check your Modification
#### 3.1 Check augmentation
直接运行tb_preprocess.py，会将augment后的图片保存在~/Pictures文件夹下
```
# Running in Terminal
cd ~/Documents/sroie2019/researches/ocr/textbox
python3 tb_preprocess.py

Output should be similar to: 
(1013, 441, 3) <= 图片输入尺寸
(512, 512, 3) <= 图片输出尺寸
/home/wang/Pictures/dataset/ocr/SROIE2019/0001.jpg cost 0.291 seconds <= 路径和所需时间
(1004, 420, 3)
(512, 512, 3)
/home/wang/Pictures/dataset/ocr/SROIE2019/0002.jpg cost 0.139 seconds
(957, 471, 3)
(512, 512, 3)
/home/wang/Pictures/dataset/ocr/SROIE2019/0003.jpg cost 0.167 seconds
(933, 436, 3)
(512, 512, 3)
/home/wang/Pictures/dataset/ocr/SROIE2019/0004.jpg cost 0.151 seconds
```

#### 3.2 Check prior box(default boxes)
取消textbox.py中函数fit里的visualize_bbox(args, cfg, image, targets, prior)的注释，便能够将可视化出来的图片保存在~/Pictures文件夹下



