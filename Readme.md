这个文档主要用于记录SA_project的代码并实现版本的管理

### 结构

project/ 

│ 

├── preprocess/

│   ├──H5ad2tsv.ipynb

│   ├──RedeconV/

│   │   └── ......

│   ├──check_H5ad.ipynb

│   └── check_tsv.ipynb



### 功能介绍

H5ad2tsv.ipynb: 实现对两个h5ad文件的转化为tsv形式（因为RedeconV归一化的输入是tsv格式）

check_tsv.ipynb和check_h5ad.ipynb: 实现对归一化前后数据检查，检验归一化之后的效果是不是好一些



