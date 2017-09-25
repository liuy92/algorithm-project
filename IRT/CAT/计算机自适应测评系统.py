#计算机自适应测评系统，Computerized Adaptive Testing（CAT）
##目前主流的计算机测评服务：TOEFL、GRE
##主要是选取最优题目组合，而不是判别学生能力
##优点：
###1. 题目量相同时，测评精度高
###2. aka自动筛选信息量最大的选题
##假设：
###1. 题目掌握程度依赖于答题结果，在允许一定程度的随机性的情况下，不同难度的题目，学霸和学渣做出来的概率不同
###2. 理论基础是Bayes理论
###3. 目标在于区分学生类型，体现学生水平（是否对学生掌握、进步存在帮助？）
##计算逻辑：
###1. 计算不同题目组合下不同题目答对状况是学霸/学渣的情况
###2. 选取能够最大程度区别学霸/学渣的可能性的题目组合为最优组合
##应用前提：
###1. 学生类型已知(用来计算不同难度题目答对率)
###2. 我们知道不同类型的学生在各个测试题上的正确率
import random

stu_layer_right = {'excellent': {'easy':0.95,'normal': 0.8,'hard': 0.6}, 'normal': {'easy': 0.9,'normal': 0.6,'hard': 0.1}}  #key用户类型(学霸、学渣), value答对率(易、中、难)
stu_degree_rate = {'excellent': 0.5, 'normal': 0.5}
def excellentChoose(df, rf, n = 2):
    user_degree = df.keys()
    user0 = df[user_degree[0]]
    user1 = df[user_degree[1]]
    rate0 = rf[user_degree[0]]
    rate1 = rf[user_degree[1]]
    degree = user0.keys()
    result = dict()
    for i, j in random.sample(degree, 2):
        xi = user0[i]
        xj = user0[j]
        yi = user1[i]
        yj = user1[j]
        result11 = rate0 * xi * xj / (rate0 * xi * xj + rate1 * yi * yj)
        result10 = rate0 * xi * (1 - xj) / (rate0 * xi * (1 - xj) + rate1 * yi * (1 - yj))
        result01 = rate0 * (1 - xi) * xj / (rate0 * (1 - xi) * xj + rate1 * (1 - yi) * yj)
        result00 = rate0 * (1 - xi) * (1 - xj) / (rate0 * (1 - xi) * (1 - xj) + rate1 * (1 - yi) * (1 - yj))
        result[(i, j)] = [result11, result10, result01, result00]
    return result

excellentChoose(stu_layer_right, stu_degree_rate)
"""
{('hard', 'normal'): [0.8888888888888888, 0.7499999999999999, 0.372093023255814, 0.18181818181818177],
 ('easy', 'normal'): [0.5846153846153846, 0.34545454545454535, 0.4000000000000003, 0.20000000000000012],
 ('hard', 'easy'):   [0.8636363636363636, 0.7500000000000002, 0.319327731092437, 0.181818181818182]}
"""


##工程设计
###
"""
学生: i
真实能力值：theta_i
对题目t预估能力值：theta_t_i
题目： j 
题目相关参数：(alpha_j, beta_j)    --常量
题目i答对题目j的正确率：
    p_i_j = exp(alpha_j * (theta_i - beta_j)) / (1 + exp(alpha_j * (theta_i - beta_j)))
这样计算的原因：Fisher information function线性可加的 + logistic模型
    p(t) = 1 / (1 + exp(-t) ) = exp(t) / (1 + exp(t))
其中，t代表学生答对这道题目，而答对题目又完全依赖于题目的相关参数 t = alpha * (theta - beta)
即若学生完成了k-1道题目，则第k道题目的Fisher信息：
    I_j(theta_i) = (p_i_j'(theta_i)) ^ 2 / (p_i_j'(theta_i) * (1 - p_i_j'(theta_i)))
"""
"""
Fisher information function(Fisher信息矩阵)
相关链接：https://www.zhihu.com/question/26561604
目标：衡量样本中的信息，用来估计分布中的未知参数
信息量的说明：认为当数据分布相对分散，方差较大，并没有集中在某个区域。样本的一阶导并不等于0，说明该数据并不是集中在某个地方
Fisher information： 
1. I(theta) = E[l'(X|theta)^2]       #即I(theta)越大，说明数据分布越不集中
2. I(theta) = - E[l"(X|theta)]       #对数似然负二阶导的期望。一阶导固定说明分布的方差固定，一阶导的分布可以反映对分布的方差估计状况，当对方差估计的越集中，说明估计效果越好。二阶导是一阶导数极值点弯曲程度的说明，
3. I(theta) = Var[l'(X|theta)]
小结：Fisher information反映的是我们对参数估计的效果的评估，它越大，则估计的越准确，所包含的信息也越多
"""
"""
MLE(最大似然估计):
思想：对预测值的估计，估计值大概率会出现在分布较高的位置。MLE的基本思想是参数会取到分布的最高位置(极值点)，则取分布一阶函数为0的答案即为参数的估计值
条件：分布已知、参数未知
似然函数：随机变量的分布已知，包含相关的参数theta未知。
          依据大数定律，独立同分布的大量样本的状况收敛于整体状况(该事情发生的实际分布)，则计算n个样本似然函数
          依据中心极限定理，其分布收敛于正态分布，则峰值(极大值)为预估参数，通过求导求解
对数似然函数：为了方便求导，对似然函数进行对数化，来求解
"""
"""
认知：总体分布实际是一种理想状态，所有可以抽到的数据均为抽取的样本，只是样本量大小的区分，统计上认为30以上的样本数量即为大样本事件
中心极限定理：
随机变量如果是大量、独立、均匀的随机变量相加组成，则其分布近似于正态分布
(例：预估某件事情的发生的可能性是多少，则多次试验，看这件事情发生的频率，每次试验独立、发生均匀，次数足够多时，这件事情发生的可能性将呈现正态分布)
大数定律：
随机事件大量、独立、均匀发生时，其发生的频率随着试验次数增加逐步收敛向一个固定值
(例：中心极限定理说明可能性的分布状况，而大数定理说明次数越多，越接近固定的值即为事件发生的概率)
1. 切比雪夫大数定律：相互独立的随机变量，方差均小于某有限固定值，随机变量抽样个数越多，样本均值逐步趋于总体均值
2. 伯努利大数定律：次数足够大，事件发生的频率越接近发生的概率
3. 辛钦大数定律：
"""

select group_id, subject, class_lv, province_id, province_name, 
    city_id, city_name, county_id, county_name, realname, disabled
from gdm.gdm_student_base_info 
where dt='2017-09-17' 
    and school_level = 1 and class_lv <=8 
    and disabled = false
    and group_id is not null
    and school_level = 1
    and realname <> '体验账号'
group by group_id, subject, class_lv, province_id, province_name, 
    city_id, city_name, county_id, county_name


select student_id, group_id, subject, class_lv,realname, disabled,school_level
from gdm.gdm_student_base_info 
where dt='2017-09-16' and student_id in (384722968,381563967,384722968,384395023,384565749,372996591,384724233,381812646,381665259,377339751,383999527,383030389,383999527,384707349)



0. 正、负样本过滤：
    正样本： 50所重点学校的老师
    负样本：(按老师的数量取)
    1. 同地区的非重点校
    2. 同地区的其他重点校
    3. 不同地区的重点校
    4. 不同地区的非重点校

现有的特征(上学期的数据：2017.03.01 - 2017.07.01；之前注册的用户)
1. 基础信息：
   1.1 老师所带班级数
   1.2 老师所带班组数
   1.3 老师所带学生数(平均数)
   1.4 是否认证

2. 活跃信息(月平均次数)：
   2.1 登陆活跃
   2.2 点击学习资源
   2.3 点击教学课件
   2.4 收到鲜花数
   2.5 登录活跃时间段(划分时间段 0:00 - 6:00； 6:00 - 10:00； 10:00 - 12:00; 12:00 - 15:00; 15:00 - 18:00; 18:00 - 0:00;计算总次数)

3. 积分行为：
   3.1 获得园丁豆(月平均数)
   3.2 消耗园丁豆(不同途径；晓丹查)

4. 行为路径：(稳定版本)
   4.1 相关点击的统计(过滤掉频度较高的，点击少量点击行为是重点关注的)
   4.2 打点顺序是否有差异(针对常用的打点，频繁项集)
   4.3 使用时间间隔(平均时间间隔)
备注：1. 统计频繁项集(高频单项删除、低频单项重点关注、高频多项观看顺序)
      2. 路径如何整合到一个老师身上(样本先按照路径来)

5. 作业数据
   5.1 布置作业月频次(assign_id)
   5.2 布置、检查作业时间段(划分时间段 0:00 - 6:00； 6:00 - 10:00； 10:00 - 12:00; 12:00 - 15:00; 15:00 - 18:00; 18:00 - 0:00;计算总次数)
   5.3 检查作业(检查率)
   5.4 作业报告分享(微信/QQ、家长通总月平均次数)
   5.5 查看作业报告(月平均次数)

6. 题目的相关状况
   6.1 布置题目的难度(教研难度、irt)[每次homework_id的平均值、标准差、中位数；然后统计到每个老师上的平均值、标准差]
   6.2 布置题目的数量(平均数、标准差)
   6.3 学生完成情况(每次作业完成率、平均正确率；平均值、标准差)
   6.4 学生能力值(irt；平均值和标准差)

一组： 1、4
二组： 0、2、3
三组： 5、 6

存表格式：学校、城市、年级、 科目、老师id、特征