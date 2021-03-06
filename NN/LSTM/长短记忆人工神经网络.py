#-*-coding:utf-8-*-

##主要学习长度那记忆单元的相关基础(LSTM)
"""
LSTM:
实质：循环神经网络
用处：识别文本,基因，笔记，语音等序列数据；识别传感器，股票等生成的数值型时间序列数据
链接：https://colah.github.io/posts/2015-08-Understanding-LSTMs/【理解】
     https://blog.csdn.net/a635661820/article/details/45390671【推导】
     https://zhuanlan.zhihu.com/p/30465140【个人认为写的比较好的论文】
相关论文：Long short-term memory in recurrent neural networks【基础】
        A Critical Review of Recurrent Neural Networks for Sequence Learning【教程】
        From Recurrent Neural Network to Long Short Term Memory Architecture Application to Handwriting Recognition Author【教程】
        LSTM: A Search Space Odyssey【对比效果改进】
        Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling【GRU】
        
"""
"""
循环网络与前馈网络：
* 前馈神经网络：神经元的信号逐层单向传递，网络无反馈，可用有向无环图表示
  也称为多层感知器，由多层的logistic回归模型组成。没有时间顺序的概念
  误差反馈：将最终误差反向按比例分配给梅阁全种，通过计算权重与误差的偏导数，利用梯度下降对权重进行调整
* 循环神经网络：神经元的信号单向循环传递，输入包含当前输入样例以及之前感知信息，有向有环图
"""
"""
循环神经网络(RNN)：
实质：增加时间序列原理，将过去的结果作为一个特征作用于现在的结果
公式： h_t = φ(W * X_t + U * h_t-1)
      其中：W为神经元的权重矩阵；X_t为当前时间的输入；h_t-1为前一事件的隐藏状态；U为隐藏状态矩阵(过度矩阵)，φ为和函数的转化
      W决定当前输入和过去隐藏状态的重要性分配。其产生的误差会反向传播返回来调整权重W
      φ 往往采用逻辑S型函数(Sigmoid函数)或双曲正切函数。将过大或者过小的值转化为一个标准数据，并产生梯度
误差反馈：BPTT(沿时间反向反馈)[表示没太看懂]

问题：梯度消失
描述：梯度表示权重随误差的变化而发生的改变，循环网络需要确定最终输出和不同时间的结果的联系。
      过往经历的重要程度难以评估，神经网络中每层之间存在大量乘积运算，更加难以分配重要度
      由于深度神经网络的层和时间步通过乘法连接，由复合利率可以发现，导数会出现膨胀或者缩减
      梯度膨胀：权重的梯度增加至饱和，导致重要性过高
      梯度缩减：权重的梯度缩减至消失，网络无法学习
解决：single sigmoids的缩胀程度最为明显；其次double sigmoids；
      然后triple sigmoids；quadruple sigmoids效果最好

问题：长期依赖(当相关信息和当前预测位置间隔增大时，会丧失学习到连接如此远的能力)
解决：LSTM可以解决(具体原理后需添加)

"""
"""
长短记忆单元(LSTM)
优势：很好的解决了梯度消失，长期依赖的问题
机制：http://www.jianshu.com/p/9dc9f41f0b29
元素：层 -- Netual Network Layer(权重)
      神经元 -- Pointwise Operation(运算法则)
      向量转化 -- Vector Transfer
      合并 -- Concatenate
差异：RNN每个模块有单一的层，而LSTM每个模块有四个神经网络层
原理：
  1.忘记门层 -- 过滤无用信息
    读取 h_t-1(上层的输出)和 x_t(当前时间的输入)，整合到 f_t = sigma(W_f * [h_t-1, x_t] + b_f)
    然后按照该模块的细胞状态C_t-1(0--完全舍弃，1--完全保留)，来决定f_t是否保留
    目的：选择是否使用旧的信息
  2.输入门层 -- 确定保留的新的信息
    通过sigmoid层决定需要更新的数据:i_t = sigma(W_i * [h_t-1, x_t] + b_i)
    通过tanh层创建新的候选集向量：tilde_C_t = tanh(W_c * [h_t-1, x_t] + b_C)
    目的：增加新的信息，决定新的信息所占的含量
  3.更新细胞状态
    旧状态同过无用信息的数据f_t相乘，丢弃掉需要丢弃的数据；
    新的候选集tilde_C_t同待更新的数据相乘i_t，确定每个状态的变化状态；
    从而得到过滤后的版本：C_t = f_t * C_t-1 + i_t * tilde_C_t
  4.确定输出信息
    确定输出的细胞状态的成分：o_t = sigma(W_o * [h_t-1, x_t] + b_o)
    将输出的细胞状态进行处理：h_t = o_t * tanh(C_t)

LSTM的变形：
  流型LSTM变体：(peephole connection)
    实质：让门层也接受细胞状态的输入
    变化：相关的[h_t-1, x_t] ——> [C_t-1, h_t-1, x_t]
          
  coupled忘记和输入门
    实质：
    变化：C_t = f_t * C_t-1 + (1 - f_t) * tilde_C_t 来取代i_t

  GRU(Gated Recurrent Unit)
  Depth Gated RNN
  Clockwork RNN
  
python包：keras、
"""
#python实现
