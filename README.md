# pysot-owndataset  
在[STVIR/pysot](https://github.com/STVIR/pysot)训练过程中加入自己的数据集  

## 代码使用场景  
1. pysot的训练过程中，除了官方提供的4个训练数据集，如果要添加自己的数据集，可参考本代码。  
2. 用于UOT100数据集的添加，同样适用于UOT32。

## 使用方法（参考原工程中VID数据集生成）  
1. 处理原UOT100数据集中命名错误的文件 `python frame2jpg.py`
2. 生成 UOT.json  `python parse_uot.py`
3. 对原图像进行剪切和目标提取 `python par_crop.py 511 12`
4. 生成训练文件 train.json 和测试文件 val.json `python gen_json.py`

## 主要修改部分  
1. 原VID数据集中包含多个subset，而UOT100中并不包含subset，所以对`parse_uot.py`中部分内容进行修改
2. 由于UOT100中标注文件是txt文件，所以对`parse_uot.py`中获取标注的方式进行修改
3. UOT100数据集中图片命名方式为`i.jpg`，不同于VID的`00000i.jpg`，对`gen_json.py`中部分内容进行修改（详见第73行-第76行）

## 运行结果  

