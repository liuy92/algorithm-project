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
ROCK -- 健壮的链接型聚类算法
实质：凝聚型层次聚类算法， 自下而上的算法
适用：分类结构的数据、高维数据；基于CURE
核心：链接 -- 确认两样本(簇)关系时考虑了他们共同邻居的数量
基本构成：
  邻居： 若两样本的相似性达到阈值 theta，则两样本为邻居
    -- theta 阈值
    -- 相似度计算： Jacard 系数， 余弦相似度
  链接： 两样本的共同邻居数量
  目标函数： 为获取最优聚类结果的度量
  相似性度量：合并两样本(簇)的依据
算法：https://www.cnblogs.com/1zhk/p/4539645.html
  1. 计算每个样本之间的相似度矩阵
  2. 依据相似度矩阵和阈值，计算邻居矩阵——若两样本之间的相似度高于阈值，则取1，否则为0 
  3. 计算链接矩阵 L = A * A。并依次计算相似性的度量，并合并相似度最高的样本
  4. 重复步骤1~3，直到形成k个聚类或聚类的数量不变
https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/rock.py

优点：
  1. 分类更鲜明，延展性很好
"""
