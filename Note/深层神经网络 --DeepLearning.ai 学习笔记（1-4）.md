>**课程笔记地址**：https://mp.csdn.net/postlist
**课程代码地址**：https://github.com/duboya/DeepLearning.ai-pragramming-code/tree/master
欢迎大家**fork**及**star**！(-^O^-)

## 1. 矩阵的维度
DNN结构示意图如图所示：

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/3.1.png)

对于第ll层神经网络，单个样本其各个参数的矩阵维度为：

- $W^{[l]}: (n^{l}, n^{[l-1]})$
- $b^{[l]}: (n^{[l]}, 1)$
- $dW^{[l]}: (n^{[l]}, n^{[l-1]})$
- $db^{[l]}: (n^{[l]}, 1)$
- $Z^{[l]}: (n^{[l]}, 1)$
- $A^{[l]} = Z^{[l]}: (n^{[l]}, 1)$

## 2. 为什么使用深度表示

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/3.2.png)

对于人脸识别，神经网络的第一层从原始图片中提取人脸的轮廓和边缘，每个神经元学习到不同边缘的信息；网络的第二层将第一层学得的边缘信息组合起来，形成人脸的一些局部的特征，例如眼睛、嘴巴等；后面的几层逐步将上一层的特征组合起来，形成人脸的模样。随着神经网络层数的增加，特征也从原来的边缘逐步扩展为人脸的整体，由整体到局部，由简单到复杂。层数越多，那么模型学习的效果也就越精确。

对于语音识别，第一层神经网络可以学习到语言发音的一些音调，后面更深层次的网络可以检测到基本的音素，再到单词信息，逐渐加深可以学到短语、句子。

所以从上面的两个例子可以看出随着神经网络的深度加深，模型能学习到更加复杂的问题，功能也更加强大。

**电路逻辑计算：**

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/3.3.png)


假定计算异或逻辑输出：

$$
y = x_{1} \bigoplus x_{2} \bigoplus x_{3} \bigoplus \cdots \bigoplus x_{1}
$$


对于该运算，若果使用深度神经网络，每层将前一层的相邻的两单元进行异或，最后到一个输出，此时整个网络的层数为一个树形的形状，网络的深度为$O(log_2(n))$，共使用的神经元的个数为：


$$
1 + 2 + \cdots + 2^{log_2(n) - 1} = 1 * \frac{1 - 2^{log_2(n)}}{1 - 2} = 2^{log_2{(n)}} - 1 = n -1
$$


即输入个数为n，输出个数为n-1。

但是如果不适用深层网络，仅仅使用单隐层的网络（如右图所示），需要的神经元个数为$2^{n-1}$个 。同样的问题，但是深层网络要比浅层网络所需要的神经元个数要少得多。


## 3. 前向和反向传播
首先给定DNN的一些参数：

- L：DNN的总层数；
- $n^{[l]}$：表示第ll层的包含的单元个数；
- $a^{[l]}$：表示第ll层激活函数的输出；
- $W^{[l]}$：表示第ll层的权重；
- 输入x记为$a^{[0]}$，输出$\hat{y}$记为$a^{[L]}。


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/3.4.png)


## 4. 参数和超参数
**参数**：

参数即是我们在过程中想要模型学习到的信息，W[l]，b[l]W[l]，b[l]。

**超参数**：

超参数即为控制参数的输出值的一些网络信息，也就是超参数的改变会导致最终得到的参数W[l]，b[l]W[l]，b[l]的改变。


举例：

- 学习速：$\alpha$
- 迭代次数：N
- 隐藏层的层数：L
- 每一层的神经元个数：$n^{[1]}, n^{[2]}, \cdots$
- 激活函数g(z)的选择


至此，完成了第一门课程的学习，感谢男神Ng的讲解！

>注：参考补充自： 
https://blog.csdn.net/koala_tree/article/details/78059952