**课程笔记地址**：https://mp.csdn.net/postlist
**课程代码地址**：https://github.com/duboya/DeepLearning.ai-pragramming-code/tree/master
欢迎大家**fork**及**star**！(-^O^-)
## 1. 神经网络的矢量化表示
![class_note_1](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/1.png)
即对于权重w，行数代表本层神经元数，列数代表本层前一层神经元数；  


对于偏差b也是同样如此，行数代表本层神经元数，列数代表本层前一层神经元数；
## 2. Pros and cons of activation functions
### 2.1 sigmoid function
除非在二分类的输出层，不然绝对不用sigmoid函数，或者直接从来不用（Ng就从来不用。因为tanh函数更方便），一般默认使用Relu函数，或者leaky Relu函数（效果更好，但一般用的也很少）。  


注：leaky value公式可能是a=max（0.01z，z）；  


注：当不确定使用哪种激活函数时候，默认使用Relu激活函数；在实际搭建神经网络时候，可以把各种激活函数都尝试一遍，通过验证集或者开发集来选出表现最好的激活函数。  
![class_note_2](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/2.png)
## 3. 初始化neural networks的权重
初始化neural networks的权重对于训练神经网络非常重要，不要直接赋值0，采用随机化赋值。（对于训练神经网络非常重要）






