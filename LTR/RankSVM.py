#-*-coding:utf-8-*-

"""
rankSVM：
实质：LTR中基于SVM的pairWise算法，将样本转化为二分类样本
流程：
1. 提取每个query下的每个document的相关度label
2. 对每个query下的任意两个document计算特征差值形成新的特征，形成新的训练实例pair
   其中若 yi > yj, 则形成新的pair：yi - yj 的label为 1， yj - yi 的label为 -1
3. 将n个文档对应的 n(n-1) 个pair构建SVM模型
4. 对新的document分别计算其与训练集中的每个document的pair，从而确定其rank


约束条件：
y * (w * (xi - xj)) < 1 - epsilon
其中 w 为构建区分超平面的参数， xi-xj为文档xi和文档xj对应的训练实例pair， y的取值为1or-1，epsilon松弛约束条件可允许误差(实现软间隔)
加入惩罚项的目标函数：
arg min(1/2*||w||^2) + C*sum(epsilon)
C为惩罚因子，衡量惩罚的程度，C越大即越忽略离群点
求解方法：拉格朗日乘子法

原因：
rankSVM是一种基于搜索的排序算法，即每个query下可供选择的document的个数并不多，故通过增加样本量来提高模型预估的精度

训练数据：每个query  每个文档   相关性(点击次数)   特征
"""