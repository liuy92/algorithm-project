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
Chameleon -- 两阶段聚类
实质：层次聚类，自下而上的算法
适用：高维数据、形状诡异的数据
核心：稀疏化 + 图划分 + 层次聚类
关键思想：仅当合并后的结果簇类似原来的两个簇时，才能合并
基本构成：
  RC： 描述两簇间边的权重是否同两簇内变得权重相近
    相对接近度 RC = S_ec(Ci, Cj) /(mi / (mi + mj) * S_ec(Ci) + mj / (mi + mj) * S_ec(Cj))
    -- mi，mj 为簇 Ci，Cj 的大小
    -- S_ec(Ci, Cj) 连接簇Ci，Cj的边的平均权值
  RI： 描述两簇间关联的边是否密集
    相对互联度 RI = EC(Ci, Cj) * 2 / (EC(Ci) + EC(Cj))
    -- EC(Ci, Cj) 链接两个簇的边的个数
  自相似性的总度量 RI(Ci, Cj) * RC(Ci, Cj) ^ alpha
    -- aplha 通常大于1
参数：
  -- k 最近邻图的k值
  -- min_size METIS时簇内最多包含的点的个数

算法：
  1. 产生k-最近邻图 (图由邻近度图导出，每个点仅包含与之相邻的k个边)
  2. METIS 二分当前最大的子图，直到不存在簇内包含多于min_size的点
  3. 凝聚层次聚类

优点：
  1. 有效剧雷空间数据
  2. 很好适应噪声和离群点
  3. 适应各种形状、密度
缺点：
  1. 划分不可逆，即产生错误不会修正
https://github.com/giovannipcarvalho/PyCHAMELEON
"""