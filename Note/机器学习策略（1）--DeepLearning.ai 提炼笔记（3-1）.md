>**课程笔记地址**：https://mp.csdn.net/postlist
>**课程代码地址**：https://github.com/duboya/DeepLearning.ai-pragramming-code/tree/master
>欢迎大家**fork**及**star**！(-^O^-)


## 结构化机器学习项目 — 机器学习策略（1）
### 1. 正交化


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/31.9.png)


表示在机器学习模型建立的整个流程中，我们需要根据不同部分反映的问题，去做相应的调整，从而更加容易地判断出是在哪一个部分出现了问题，并做相应的解决措施。

正交化或正交性是一种系统设计属性，其确保修改算法的指令或部分不会对系统的其他部分产生或传播副作用。 相互独立地验证使得算法变得更简单，减少了测试和开发的时间。

当在监督学习模型中，以下的4个假设需要真实且是相互正交的：

- 系统在训练集上表现的好 
	- 否则，使用更大的神经网络、更好的优化算法
- 系统在开发集上表现的好 
	- 否则，使用正则化、更大的训练集

> 因为这种情况的发生往往是因为你overtune了你的Dev set，这时候通过换用更大的dev set会有利于模型优化。

- 系统在测试集上表现的好 
	- 否则，使用更大的开发集
- 在真实的系统环境中表现的好 
	- 否则，修改开发测试集、修改代价函数
### 2. 单一数字评估指标
在训练机器学习模型的时候，无论是调整超参数，还是尝试更好的优化算法，为问题设置一个单一数字评估指标，可以更好更快的评估模型。

![](![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/31.1.png))

注意：上图中讲明白了precision、Recall的由来，注意自己老是将Recall理解错！
此外，accuracy的定义对于给定的测试数据集，分类器正确分类的样本数与总样本数之比。也就是损失函数是0-1损失时测试数据集上的准确率
accuracy = (True positive + True negative) / (True positive + True negative + False positive + False negative)

#### example1
下面是分别训练的两个分类器的Precision、Recall以及F1 score。

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/31.1.png)

由上表可以看出，以**Precision**为指标，则分类器A的分类效果好；以**Recall**为指标，则分类器B的分类效果好。所以在有两个及以上判定指标的时候，我们很难决定出A好还是B好。

这里以Precision和Recall为基础，构成一个综合指标**F1 Score**，那么我们利用**F1 Score**便可以更容易的评判出分类器A的效果更好。

指标介绍：

在二分类问题中，通过预测我们得到下面的真实值$y$和预测值$\hat{y}$的表：


