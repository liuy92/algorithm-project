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