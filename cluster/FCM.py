#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
FCM -- Fuzzy C-means -- 模糊C均值 -- ISODATA
实质：基于原型的、
核心：模糊集合论 + 模糊逻辑
基本概念：
  1. 模糊簇：簇内任意点在各簇的权值之和为1；各簇内至少包含一个非零、非1的点
  2. 隶属度：点属于某集合的可能性，取值范围[0, 1]
  3. 模糊伪划分：包含模糊簇的数据划分
特点：
  1. 数据归一化后使一个数据记得隶属度和为1 
  2. 目标函数：误差平方和SSE = sum(sum(wij ^ p * dist(xi, cj) ^ 2))
参数：
  1. 分类个数： C
  2. 类别的权重：wij
  3. 权值影响参数 p  (p越接近1， 越接近K-means；p越大，划分越模糊)
算法：http://blog.csdn.net/on2way/article/details/47087201
  1. 随机初始化C个模糊伪划分，并随机给予每个类别权重(与任何对象向关联的权重值和必须为1)[初始化隶属矩阵]
  2. 计算每个簇的质心 Cj = sum(wij ^ p * xi) / sum(wij ^ p)
  3. 更新簇的权值：(最小化SSE得出)wij = (1 / dist(xi, cj) ^ 2) ^ (1 / (p - 1)) / sum((1 / dist(xi, cj) ^ 2) ^ (1 / (p - 1)))
  4. 重复步骤2、3，直到质心不发生变化
特点：
  1. 适合球形的数据
  2. 对异常值敏感
  3. 计算复杂度较高

https://pypi.python.org/pypi/pyfcm/
https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
"""