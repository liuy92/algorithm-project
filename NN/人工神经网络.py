#-*-coding:utf-8-*-

#https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
#https://deeplearning4j.org/cn/neuralnet-overview
#https://nndl.github.io/ch5.pdf
#归属NLP（Natural Language Processing）自然语言处理
"""
神经网络模仿人脑设计，用于识别模式。主要用来聚类和分类。
模式：神经网络提取特征；算法聚类分类
条件：有监督的学习(在明白需求前提下，将有类别的样本作为训练集进行模型计算)
构成：节点-->多层堆叠
  1. 节点：数据 + 权重 + 激活函数 = 信号传递 + 传递距离 控制相应特征在算法中的重要程度
  2. 层级：由多个节点组成：输入层 + 隐藏层 + 输出层。包含一个以上隐藏层即深度学习
实质：层级见的特征传递。每层即将上一层输入的数据特征经过一定整合处理，输出到下一层作为特征继续传递
PS：重构的方式--玻尔兹曼机
从概念上讲：深度学习 < 机器学习 < 人工智能(AI)

主要分类：
  1. 感知器
  2. 前馈网络
  3. 卷积网络
  4. 循环网络
  5. 自组织映射
  6. HopField网络
  7. Boltzmann机
备注：以上网络的主要差异在于神经元的激活机制以及网络连接的拓扑结构、参数的学习规则(调优方式)不同

调整准则：
  1. 输入 * 权重 = 猜测值
  2. 实际基准 - 猜测值 = 误差
  3. 误差 * 权重于误差的影响 = 调整幅度
机制： 输入特征 --> 计算误差 --> 更新模型 ；然后不断迭代(正性反馈迭代：奖励则增加权重，惩罚即降低权重)
Tips：
  1. 网络的拓扑结构：每层的计算基础是多元线性回归，权重即系数；
  2. 最后一层分类，输出层往往使用逻辑回归
  3. 神经元的激活规则：为例更好表达非线性，层级的回归基础上增加激活函数
  4. 为了更好的迭代和调参，评估准则是梯度下降，而其中的激活值的计算方式即为更新器
多层网络类型：
  1. 堆叠式降噪自动编码
  2. 深度置信网络


"""
#学习基础
"""
*线性回归(层级特征转化)
*梯度下降(用作权重优化)
*更新器
*激活函数
*逻辑回归
"""
"""
梯度下降：
实质：梯度即斜率；降低权重带给误差的影响 dE / dw ；衡量权重调整后误差的变化程度
问题：完整的神经网络中包含大量的节点，极多的权重。权重的信号会经过多层的激活和求和运算。
      则得到总体误差和权重之间的关系，依据微积分的链式法则：
                  dz / dx = dz / dy * dy / dx
      dy 即激活值作为中介传递
"""
"""
神经网络工具包：DL4J
https://zhuanlan.zhihu.com/p/22252270
语言：基于Java构建，与Hadoop和Spark集成
支持的更新器(迭代相关参数的方法)：
  1. AdaDelta()
  2. AdaGrad()
  3. Adam()
  4. Nesterov()
  5. NONE()
  6. RMSprop()
  7. SGD(mini-batch gradient descent)
  8. Conjugate Gradient()
  9. Hessian Free()
  10.LBFGS()
  11.Line Gradient Descent()
其他更新器：
  1. Momentum
  2. Adamax
  3. Nadam

激活函数：增加非线性因素(用来帮助处理线性不可分的数据)。将数据转化为易识别的格式
          用以每层叠加完成后，帮助增加神经网络的表达性
          实质即把激活的神经元的特征(每层经过筛选的特征)进行进一步的映射和保留。从而帮助将权值叠加结果转化为分类结果
DL4J支持的激活函数：
  1. Cube
  2. ELU
  3. HardSigmoid
  4. Identity
  5. LeakYrelu
  6. RationalTanh
  7. Relu
  8. RRelu
  9. Gigmoid
  10.SoftMax
  11.SoftPlus
  12.SoftSign
  13.Tanh

DL4J支持的损失函数：
  1. MSE：均方差
  2. Expll：指数对数似然函数
  3. Xent：叉熵二元分类
  4. MCXent：多类别叉熵
  5. RMSE_Xent：RMSE叉熵
  6. SquaredLoss：平方损失
  7. NegativeLoglikehood：负对数似然函数
"""
"""
评估参数：
  1. 学习速率(步幅)：函数在搜索的速率。一般范围：0.001~0.1之间。步幅越小，结果越精确
  2. 动量：是矩阵变化率的导数的因数。决定向最优值收敛的速度。一般范围：0~1之间。动量越大，定性速度越快，但模型准确率可能会降低
  3. L2正则化常数：主要针对逻辑回归的街而过的评估
"""
"""
问题：
  1. 对有监督学习，标签(类别)数量和数据量之间的关系：标签量越大，计算强度越大
  2. 若按批次输入数据，每个批次的数据量应该是多少：批次越大，更新间的等待时间(学习步骤)就越长。初次往往批次在1000次左右
  3. 特征量状况：特征量越多，所需内存越大(特征越多，往往要求神经网络的构建层数越多，内存所占也就越大)
  4. 调试方法：1）观察网络的F1值来调节超参数；2）使用超参数优化工具实现自动化调试[https://github.com/deeplearning4j/Arbiter]
  5. 所需硬件：研究所需往往是1~4个GPU系统。而企业级的计算需要使用大型CPU集群
  6. 关于特征提取：事先的特征工程能够有效的减轻计算负荷，加快定性速度，提高效率，尤其是在数据稀疏的情况下
  7. 优化的目标：误差函数、目标函数、损失函数(目的均为最小化误差)
"""
#python包
"""
keras:  https://keras.io/；http://www.360doc.com/content/17/0624/12/1489589_666148811.shtml
模块：
  1.optimizers
    用途：选择优化方法
    1.1 SGD
    1.2 AdaGrad
    1.3 AdaDelta
    1.4 RMSprop
    1.5 Adam
  2.objectives 
    用途：用来优化的误差项选择
    2.1 mse：均方差(mean_squared_error)
    2.2 mae：绝对误差(mean_absolute_error)
    2.3 msle：对数误差(mean_absolute_percentage_error)
    2.4 mape：评价绝对百分差(mean_squared_logarithmic_error)
    2.5 squared_loss：平方损失
    2.6 hinge
    2.7 binary_crossentropy
    2.8 categorical_crossentropy
  3.model
    用途：建立神经网络的一般操作
    3.1 Sequential() 初始化一个网络
    3.2 add 添加一层神经网络
    3.3 compile 
  4.layers
    用途：给搭建的神经网络增加网络层
    4.1 Dense
    4.2 Dropout
    4.3 Activation
    4.4 Flatten
    4.5 Convolution2D 卷积层
    4.6 Maxpooling2D 卷积层
"""