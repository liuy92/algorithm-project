#-*-coding:utf-8-*-

#主要讲解聚类的适用范围以及聚类效果评估以及相关参数的选择
"""
聚类
目的：发现从概念意义上存在共性的数据对象，并讲这些对象划分为不同的组别，针对不同的对象进行相关的定义和操作
应用：
1. 创建体系、层次结构
2. 信息数据合并，方便检索
3. 查询结果产生的原因，并按照原因来改善结果 (运营、治疗)
4. 刻画特征

聚类算法的分类
  结果导向的分类：
    1. 层次分类(嵌套)
    2. 划分分类(未嵌套)
    3. 互斥聚类
    4. 重叠聚类
    5. 模糊聚类

  数据导向的分类(簇的结构)
    1. 明显分离的 —— 任意形状，类别明显
    2. 基于原型的 —— 计算距离，确定质心，球形分类
    3. 基于图的   —— 连通分支，边定权重，噪声严重
    4. 基于密度的 —— 分布密度，适宜噪声，形不规则
    5. 共同性质的 —— 强依主管，需要先验，概念簇弃
说明：https://www.zhihu.com/question/34554321
"""

"""
数据处理：
1. 数据标准化，减少量纲不同带来的作用
2. 针对不同的聚类方法，减少维度影响
3. 按照聚类算法特性，决定是否删除异常值点
"""

"""
适用条件：样本在空间内分布尽量足够聚集，而不是均匀的分布
  检测：检测样本是否分布均匀——霍普金斯统计量(Hopkins Statistic)
  步骤：1. 从样本空间D对应的总空间中均匀、随机抽取n个点，每个点找到其相应在样本空间中的最近邻：x_i = min(dist(p_i, v))
        2. 均匀的从样本空间D中抽取n个点，对每个点找到其在除抽的点外其他点的最近邻：y_i = min(dist(q_i, v))
        3. 计算Hopkins Statistic： H = sum(Y) / (sum(Y) + sum(X))
  结论：H越接近0，说明D的数据具有很高的倾斜性，数据越聚集；若H越接近0.5。说明数据分布很均匀
"""
"""
确定聚类的个数(簇数)：http://stat.smmu.edu.cn/field/sas07.html
  1. 经验判断： k = sqrt(n / 2)
  2. 肘方法：分别统计不同簇数对应的总体的离差平方和，其分布持续下降，但斜率先升后降，选择斜率最大的位置对应的簇数
  3. PSF：F = ((T - P_G) / (G - 1)) / (P_G / (n - G))
  4. PST2:t = B_KL / (() + ())  取其最大值对应簇数 + 1
  5. 信息论、信息准则
  6. 交叉验证

备注：主要选择的依据为离差平方和(求样本的中心点，在计算每个样本点到中心点距离的平方和；若存在多累，分别计算每个簇的离差平方和，累加所有簇的离差平方和)
"""
"""
评估聚类质量：
  1. Bcubed(外在方法)：包含精度和召回率
  2. 轮廓系数(内在方法)：计算每个簇里面每个点和其他点的平均距离的平均值a(o)；在计算其他两个簇里面的所有点的平均距离b(o);轮廓系数：s(o) = (b(o) - a(o)) / max(a(o)， b(o)),其值在-1到1之间，越接近1越好
"""
"""
一般聚类算法：http://scikit-learn.org/stable/modules/clustering.html
进阶聚类算法：
  Pythoon 包：pyclustering
  链接：https://pypi.python.org/pypi/pyclustering
  源代码：https://github.com/annoviko/pyclustering
  主要算法  
    Agglomerative (pyclustering.cluster.agglomerative);
    BIRCH (pyclustering.cluster.birch);
    CLARANS (pyclustering.cluster.clarans);
    CURE (pyclustering.cluster.cure);
    DBSCAN (pyclustering.cluster.dbscan);
    EMA (pyclustering.cluster.ema);
    GA (Genetic Algorithm) (pyclustering.cluster.ga);
    HSyncNet (pyclustering.cluster.hsyncnet);
    K-Means (pyclustering.cluster.kmeans);
    K-Means++ (pyclustering.cluster.center_initializer);
    K-Medians (pyclustering.cluster.kmedians);
    K-Medoids (PAM) (pyclustering.cluster.kmedoids);
    OPTICS (pyclustering.cluster.optics);
    ROCK (pyclustering.cluster.rock);
    SOM-SC (pyclustering.cluster.somsc);
    SyncNet (pyclustering.cluster.syncnet);
    Sync-SOM (pyclustering.cluster.syncsom);
    X-Means (pyclustering.cluster.xmeans);
"""