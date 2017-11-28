#-*-coding:utf-8-*-
import sys

reload(sys)
sys.setdefaultencoding("utf8")

"""
该脚本主要讲解分层聚类的相关算法
主要路径：
1. 自上而下 -- top_down
2. 自下而上 -- bottom-up

聚类原则：
1. 最短距离法
2. 最远距离法
3. 中间距离法
4. 类平均法
备注：类平均法应用最广泛，原因是单调性和空间收缩和空间扩张性适中

主要算法：
1. BIRCH -- Balanced Iterative Reducing and Clustering Using Hierarchies
2. ROCK -- A Hierarchical Clustering Algorithm fir Categorical Attributes
3. Chameleon -- A Hierarchical Clustering Algorithm Using Dynamic Modeling
"""

"""
BIRCH -- 利用层次方法的平衡迭代规约和聚类
实质：增量算法, 自上而下的算法
适用类型：数据量大、数据类型为数值型
核心：CF树
特点：
1. 有限的内存，最小化I/O时间
2. 多阶段聚类技术：单边扫描生成基本的聚类，多遍的额外扫描来优化改进聚类质量
3. 增量算法：每个聚类决策基于处理过的数据，不是全局聚类，依赖局部聚类原则
4. 可识别噪声点
5. 依据直径空值聚类边界，只适用于球形的数据结构
6. 对高维数据的聚类效果不好
7. 聚类效果强依赖于B和L
基本构成：
  N个d维数据点 {x1, x2, x3, ..., xd}
  每个簇对应的 CF = (N, LS, SS) 
    -- N 簇的数据点个数
    -- LS 簇内N个节点的线性和
    -- SS N个节点的平方和
    簇的质心： C = (x1 + x2 + ... + xn) / n
    簇的半径： R = (|x1 - C| ^ 2 + ... + |xn - C| ^ 2) / n  [簇内所有数据点到质心的平均距离]
  聚类特征树： CF Tree
    参数： 内部节点平衡因子B， 叶节点平衡因子L， 簇半径阈值T
    -- B 树中根节点最多包含的叶子节点， 每个节点实质就是一个簇 (CFi, CHILDi)
    -- L 树中叶子节点最多包含的孩子节点
算法：https://www.cnblogs.com/pinard/p/6179132.html
  1. 创建空的CFTree，读入第一个数据点，形成一个CF
  2. 加入第二个数据点，按照半径阈值，判断是否形成新的CF
  3. 重复步骤2，直到加入第i个数据点，根节点包含B个CF，且i不在这B个CF中任意一个中的半径T中。则分裂根节点为两个叶子结点。并分别计算每个CF到根节点的距离，将距离远的CF放在一个叶子节点中。
  4. 重复步骤3，直到加入第j个数据点，属于叶子结点LN，该叶子节点包含L个CF，且j不属于L个CF。则分裂叶子结点为两个叶子结点。
  5. 直到完成全部分裂
"""
from sklearn.cluster import Birch
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3], random_state =9)
##设置birch函数
birch = Birch(n_clusters = None)
##训练数据
y_pred = birch.fit_predict(X)
##绘图
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
