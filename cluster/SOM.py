#-*-coding:utf-8-*-
import sys

reload(sys)
sys.setdefaultencoding('utf8')

"""
SOM -- 自组织特征映射
实质：基于原型的，层次聚类
目标：用低维目标空间的点表示高维空间的所有点
核心：神经网络 + 地形顺序
参数：
  1. 初始质心个数 k
  2. 距离度量 欧几里得距离、点积度量、余弦度量
  3. 网格类型 不同网格类型，对质心的计算方法不同质心的数量也会发生变化  m_j(t + 1) = m_j(t) + h_j(t) * (p(t) - mj(t))
  4. 邻域函数 h_j(t) 确定分类的半径，随着迭代次数的增加，邻域函数的半径应该是收缩的 h_j(t) = alpha(t) * exp(-dist(rj, rk) ^ 2 / (2 * sigma(t) ^ 2))
  5. 学习率参数 aloha(t) [0, 1] 随时间单调减少，控制收敛率
算法：
  1. 向量归一化 + 随机指定 k 个神经元(质心)对应的权重向量
  2. 依次计算每个样本到各个神经元(质心)的距离(相似度)，然后将样本归于距离较近的神经元后，更新神经元的权重向量(邻域函数 + 网格类型)
  3. 循环步骤2，收缩邻域半径、减小学习率，直到质心收敛或者变化很小

优点：
  1. 相邻关系强加在簇质心上，可以评估不同簇之间的关联程度
  2. 聚类结果容易可视化和解释度较高
缺点：
  1. 含参数较多
  2. 目标函数不明确
  3. 通常不对应于单个自然簇。当其密度、形状、大小存在某一点不一致，则结果会合并或分裂
  4. 不保证绝对收敛
  5. 不好处理缺失值的数据
  6. 计算复杂度较高，不适合海量数据

example : http://nbviewer.jupyter.org/gist/sevamoo/f1afe78af3cf6b8c4b67
"""
import numpy as np
from sompylib.som_structure as SOM
from matplotlib import pyplot as plt

dim = 3
data = np.random.randint(0, 2, size = (100 * 1000 * 1 * 1 * 1, dim))
reload(sys.modules['sompylib.som_structure'])
sm = SOM.SOM('sm', data, mapsize = [50, 50], norm_method = 'var', inirmethod = 'pca')
sm.train(n_job = 2, shared_memory = 'no')

tmp = np.zeros((msz0,msz1,dim))
codebook = getattr(sm,'codebook')
codebook = SOM.denormalize_by(Data,codebook)
# codebook = SOM.denormalize(Data, codebook)
for i in range (codebook.shape[1]):
    tmp[:,:,i] = codebook[:, i].reshape(msz0,msz1)
from matplotlib import pyplot as plt
tmp.shape
fig = plt.imshow(tmp[:,:,0:3])







