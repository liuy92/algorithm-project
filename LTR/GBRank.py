#-*-coding:utf-8-*-

"""
GBRank 
实质：基于GBDT的pairWise的LTR
基本思想：对两个具有相同的relative  relevance judgment (相同的query下)的document，构建相应pair后，搭建相关的loss function，并用GBDT求解出最优值，构建得到相应分类器
损失函数：1/2 sum(max(0, dao - (yi1_hat - yi2_hat))^2)


"""