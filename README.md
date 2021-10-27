# pysot-owndataset  
在[STVIR/pysot](https://github.com/STVIR/pysot)训练过程中加入自己的数据集  

## 代码使用场景  
1. pysot的训练过程中，除了官方提供的4个训练数据集，如果要添加自己的数据集，可参考本代码。  
2. 用于UOT100数据集的添加，同样适用于UOT32。

## 使用方法（参考原工程中VID数据集生成）  
1. 处理原UOT100数据集中命名错误的文件 `python frame2jpg.py`
2. 生成 UOT.json  `python parse_vid.py`
3. 对原图像进行剪切和目标提取 `python par_crop.py 511 12`
4. 生成训练文件 train.json 和测试文件 val.json `python gen_json.py`

## 主要修改部分  
1. 
