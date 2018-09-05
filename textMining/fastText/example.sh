#主要记录如何使用fasttext构建词向量和进行文本分类,存在一些参数的设置详见example.py

cd fastText  #进入安装目录（必须make后才能进行下一步）

#构建cbow_model;data.txt的格式为每行为一个样本，每个单词之间用空格分割；输出包含两部分：cbow_model.bin存储模型基本内容，可以load；cbow_model.vec输出词向量

./fasttext cbow -input data_path/data.txt -output model_path/cbow_model 

#构建skipgram模型；data.txt格式相同；输出部分相同

./fasttext skipgram -input data_path/data.txt -output model_path/skgm_model 

#新数据集text.txt的预估(skipgram模型相同，用cbow模型结果做示范)

cat text.txt | ./fasttext print-word-vectors cbow_model.bin

./fasttext print-word-vectors cbow_model.bin < text.txt

#文本分类
