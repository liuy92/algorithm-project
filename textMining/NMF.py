#-*-coding:utf-8-*-
import sys

reload(sys)
sys.setdefaultencoding('utf8')

#协同过滤NMF算法
"""
NMF (non-nefative matrix factorization)
非负矩阵分解： V_fn ~ W_fk * H_kn
    W 权重矩阵
    H 特征矩阵
    V 原矩阵
    f 样本个数
    n 特征个数

条件：W、H所有元素均大于0

"""