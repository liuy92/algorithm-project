#-*-coding:utf-8-*-
import sys

reload(sys)
sys.setdefaultencoding('utf8')

"""
AP Clustering -- Affinity Propagation -- 近邻传播算法
实质：自下到上的聚类
核心：
目标函数：
基本构成：
  1. 参考度 -- preference：相似度矩阵对角线上的元素，代表对应样本是否适合作为簇的质心，其个数从一定程度上讲可以决定聚类的个数
  2. 吸引度 -- responsibility：某样本点是否能够作为其他样本的质心
  3. 归属度 -- availability：除本身外，某质心是否合适作为其他所有点的质心
  4. 阻尼系数 -- Damping factor： 衰减系数，delta * information(t - 1) + (1 - delta) * information(t) [0, 1], default -- 0.5

参数：
  1. 参考度
  2. 阻尼系数

算法：
  1. 按照 r(i, k) = s(i, k) - max(a(i, k') + s(i, k')) 计算每个样本对其他样本的初始的吸引度[k 对 i的吸引力是其本身的相似度 减去 其他点对 i 的最大诱惑]
  2. 按照 a(i, k) = min(0, r(k, k) + sum(max(0, r(i', k)))) 计算每个样本的归属度[k 对所有点的吸引度和]
  3. 按照 r_t(i, k) = (1 - delta) *(s(i, k) - max(a(i, k') + s(i, k'))) + delta * r_t-1(i, k) 更新吸引度
  4. 按照 a_t(i, k) = (1 - delta) * min(0, r(k, k) + sum(max(0, r(i', k)))) + delta * a_t-1(i, k) 更新归属度
  5. 计算吸引度和归属度的和 sum
  6. 迭代步骤3、4、5，直到sum不再发生较为显著的变化(收敛)，停止
  
优点：
  1. 不需要确定最终聚类个数，但需要确定参考度
  2. 不会新生成簇心，以数据点作为簇心
  3. 对初始值不敏感
  4. 比K-means的平方差误差小

缺点：
  1. 算法复杂度高，计算较慢，不适宜海量数据
"""
from sklearn.cluster import AffinityPropagation   
from sklearn import metrics   
from sklearn.datasets.samples_generator import make_blobs   
import numpy as np   
  
# 生成测试数据   
centers = [[1, 1], [-1, -1], [1, -1]]   
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)   
  
  
# AP模型拟合   
af = AffinityPropagation(preference=-50).fit(X)   
cluster_centers_indices = af.cluster_centers_indices_   
labels = af.labels_   
new_X = np.column_stack((X, labels))   
  
n_clusters_ = len(cluster_centers_indices)   
  
print('Estimated number of clusters: %d' % n_clusters_)   
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))   
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))   
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))   
print("Adjusted Rand Index: %0.3f"   
      % metrics.adjusted_rand_score(labels_true, labels))   
print("Adjusted Mutual Information: %0.3f"   
      % metrics.adjusted_mutual_info_score(labels_true, labels))   
print("Silhouette Coefficient: %0.3f"   
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))   
print('Top 10 sapmles:',new_X[:10])   
  
# 图形展示   
import matplotlib.pyplot as plt   
from itertools import cycle   
  
plt.close('all')   
plt.figure(1)   
plt.clf()   
  
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')   
for k, col in zip(range(n_clusters_), colors):   
    class_members = labels == k   
    cluster_center = X[cluster_centers_indices[k]]   
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')   
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,   
             markeredgecolor='k', markersize=14)   
    for x in X[class_members]:   
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)   
  
plt.title('Estimated number of clusters: %d' % n_clusters_)   
plt.show()   