#-*-coding:utf-8-*-
#user:me
#date:20180905
#describe:操作使用fasttext的例子；见 https://pypi.org/project/fasttext/
"""
输入参数：
input_file     训练数据的路径及文件名（必填）
output         输出结果，常填 model（必填）
lr             学习率 [0.05]
lr_update_rate 更新学习率次数 [100]
dim            输出词向量的长度 [100]
ws             窗口宽度or每次进行估测的周围词的个数 [5]
epoch          number of epochs [5]
min_count      minimal number of word occurences [5]
neg            number of negatives sampled [5]
word_ngrams    最长的ngram的单词个数 [1]
loss           损失函数选择有三种，用来进行梯度下降 {ns, hs, softmax} [ns]
bucket         桶数 [2000000]
minn           最短字符长度（感觉和词缀拆分有关） [3]
maxn           最长字符长度 [6]
thread         number of threads [12]
t              阈值 [0.0001]
silent         disable the log output from the C++ extension [1]
encoding       encoding格式 [utf-8]

输出api：
model.model_name       # Model name
model.words            # List of words in the dictionary
model.dim              # Size of word vector
model.ws               # Size of context window
model.epoch            # Number of epochs
model.min_count        # Minimal number of word occurences
model.neg              # Number of negative sampled
model.word_ngrams      # Max length of word ngram
model.loss_name        # Loss function name
model.bucket           # Number of buckets
model.minn             # Min length of char ngram
model.maxn             # Max length of char ngram
model.lr_update_rate   # Rate of updates for the learning rate
model.t                # Value of sampling threshold
model.encoding         # Encoding of the model
model[word]            # Get the vector of specified word
"""


import fasttext
#构建相应模型
cbow_model = fasttext.cbow('{}/data/accurate_interest_all_vector.txt'.format(path), 'model')
skgm_model = fasttext.skipgram('{}/data/accurate_interest_all_vector.txt'.format(path), 'model')
#输出模型的词向量
for i in cbow_model.words:
    cbow_output.write(i + '\t' + '\t'.join(cbow_model[i]) + '\n')
cbow_output.close()
for i in skgm_model.word:
    skgm_output.write(i + '\t' + '\t'.join(skgm_model[i]) + '\n')
skgm_mdodel.close()

dir(cbow_model)
