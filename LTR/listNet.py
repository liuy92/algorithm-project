#-*-coding:utf-8-*-

"""
rankNet
http://blog.csdn.net/puqutogether/article/details/42124491
http://blog.csdn.net/puqutogether/article/details/43667375
实质：基于神经网络的pairWise的LTR
数据：输入：不同query对应的所有文档特征，每个query下的文件需要人为标注得分，得分越高说明排名应该越靠前
      输出：打分模型，得出每个文档在相应的query下的打分
相关概念：
    1. 对任意两个document会存在得分组(si, sj) -- s为模型计算的document的得分
    2. document i排在document j前面的概率p_ij = (1 + exp(-sigma * (si - sj))) ^ -1
       这里用的是sigmoid函数的原因是，si、sj为得分，当si得分越高于sj，说明document i越应该排在document j前面，即概率月接近1，反之越接近0
    3. 真实概率定义函数 Sij = 1 or -1 or 0，则document i和document j的更靠前的真是概率 p_ij = (1 + Sij) / 2
    4. 损失函数按照 document 对的排序关系的预测准确与否。
       对每个document 对(pair)的交叉熵：Cij = -pij*log(pij_hat) - (1-pij)*log(1-pij_hat)
       Cij = log(1 - exp(si - sj)) + sigma * (1- sij) * (si - sj) / 2
    5. 得分的计算(三层排序函数的定义)：si = g(sum(wij * g(sum(wjk * xnk + bj ^ 2)) ^2 + bi ^ 2)) ^ 3
损失函数性质：
    1. 若两document的分数相同，认为其真实排序为相同时，仍然会产生损失值 —— 该模型会尽量给两个document不同的分数
    2. 损失函数在排序一定的情况下，是单调的，具有稳健性，由于类线性分布，异常值作用小
    3. 总损失函数 C = sum(cij)
    4. 核心：梯度下降算法，来估计参数w

备注：
1. rankNet实质也是对document对来做预测模型，不同于pairWise的是其样本是组合数，而pairWise是排列数
   rankNet是缩小损失函数来优化排序，而pairWise是构建二分类模型来估计大小
   rankNet的输出是分数值，而pairWise输出值是谁大谁小
2. rankNet是三层神经网络，第一层输入每个document的特征，到第2层计算某个样本特征加权+偏置后的输出值。第三层是对特征加权的和经过各个节点的整合后输出这个样本的分数
3. 只关注两两差异的pair对，而不是关注全局排序的结果

LambdaRank
实质：基于NDGG、ERR等损失函数的ListWise的LTR



listNet
http://x-algo.cn/index.php/2016/08/18/listnet-principle/
实质：基于神经网络的ListWise的LTR
数据：搜索条件query
      每个query对应的检索结果：di
      每个检索结果对应得分(相关度)：yi
      损失函数：sum(L(yi,f(di))) = -sum(yi * log(yi_hat))  --交叉熵损失
主要思想：将所有样本的排序看作是一个pair，
          1. 每个排序出现的概率为[0,1]
          2. 所有排序的概率和为1
          3. 可以计算每个元素分别处在第l位的概率：exp(yil) / sum(exp(yij))
          4. 可以计算每个yi，预估为yi_hat的损失函数：L(yi, yi_hat) = -sum(yi * log(exp(yil) / sum(exp(yij)))    -- l in [1,n]

流程：
1. 提取数据，输入相关参数：m - 随机抽取query的个数，T - 重复建模次数，fi - 学习率
2. 重复更新T次，分别计算每个query中在NN中所有排序对应损失函数的梯度差异
3. 并按照所给定的学习率来更新w，来构建NN
"""