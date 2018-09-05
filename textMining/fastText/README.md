fastText
开发：Facebook
目的：构建词向量、预估上下文、文本分类
github链接：https://github.com/facebookresearch/fastText
相关链接：https://blog.csdn.net/sinat_26917383/article/details/54850933  http://www.52nlp.cn/fasttext
包含内容：模型架构、层次softmax、N-gram
实质：叠加n-gram词汇作为softmax的特征进行分类
思路：在预估词汇时，选择中心词周围的词汇作为特征词组，在投影层将特征词组的整合or展开。在投影层通过树模型 + softmax模型，最终输出中心词词向量。而中间的权重参数or词向量均通过构建相应的loss函数，然后通过梯度上升or梯度下降求解；树模型的构建依据是按照词频来计算，可以快速的检索到高频词汇，加快预测效率
主要工具：CBOW、Skipgram、
改良：存在涉及字符级别的元素拆分（主要针对英文中将apple 拆分成ap ple）
底层语言：C
优点：
1. 对大量数据的高效率
2. 可支持多种语言：C、C++、python


安装：
1. python
   pip install fasttext
2. zip解压
   wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
   unzip v0.1.0.zip
   cd fastText-0.1.0
   make
3. git拷贝
   git clone https://github.com/facebookresearch/fastText.git
   cd fastText
   make && make install


