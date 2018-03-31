# 神经网络和深度学习—神经网络基础
## 1.二分类问题
对于二分类问题，Ng给出了小的Notation。
- 样本：(x,y)，训练样本包含m个；
- 其中$x \in R^{n_x}$，表示样本x包含$n_x$个特征；
- $y \in 0,1$，目标值属于0,1分类；
- 训练样本数据：$\{(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)}), ... ,(x^{(m)},y^{(m)}) \}$；

![3](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/3.png)

X.shape = ($n_x$, m)
目标数据的形状：
Y = [$y_(1), y_(2), ..., y(m)$]
Y.shape = (1,m)

## 2. logistic Regression

![4](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/4.png)

## 3. logistic Regression loss function

**Loss  function**
![5](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/5.png)

**Cost function**
![6](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/6.png)

## 4. gradient descent
![7](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/7.png)

## 5. gradient descent of logistic regression
![8](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/8.png)

**反向传播过程**
![9](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/9.png)

## 6. m个样本的梯度下降
![10](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/10.png)

## 7. 向量化（vectorization）
在深度学习的算法中，我们通常拥有大量的数据，在程序的编写过程中，应该尽最大可能的少使用loop循环语句，利用python可以实现矩阵运算，进而来提高程序的运行速度，避免for循环的使用。
**vectorization of logistic regression**

- 输入矩阵X: ($n_x, m$)

- 权重矩阵w: ($n_x, 1$)

- 偏置b: 为一个常数

- 输出矩阵Y: (1, m)
所有m个样本的线型输出Z可以用矩阵表示：
$$Z = w^TX + b$$
python代码：
```py
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
```

![12](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/12.png)

**单词迭代梯度下降算法流程**

```py
Z = np.dot(w.T, X) + b

A = sigmoid(Z)

```

![13](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/13.png)

**单次迭代梯度下降算法流程**
```
Z = np.dot(w.T,X) + b
A = sigmoid(Z)
dZ = A-Y
dw = 1/m*np.dot(X,dZ.T)
db = 1/m*np.sum(dZ)

w = w - alpha*dw
b = b - alpha*db
```
## 8. python的notation

- 虽然在Python有广播的机制，但是在Python程序中，为了保证矩阵运算的正确性，可以使用reshape()函数来对矩阵设定所需要进行计算的维度，这是个好的习惯；

- 如果用下列语句来定义一个向量，则这条语句生成的a的维度为（5，），既不是行向量也不是列向量，称为秩（rank）为1的array，如果对a进行转置，则会得到a本身，这在计算中会给我们带来一些问题。

```
a = np.random.randn(5)
```

- 如果需要定义（5，1）或者（1，5）向量，要使用下面标准的语句：

```
a = np.random.randn(5,1)
b = np.random.randn(1,5)
```

- 可以使用assert语句对向量或数组的维度进行判断。assert会对内嵌语句进行判断，即判断a的维度是不是（5，1），如果不是，则程序在此处停止。使用assert语句也是一种很好的习惯，能够帮助我们及时检查、发现语句是否正确。

```
assert(a.shape == (5,1))
```

- 可以使用reshape函数对数组设定所需的维度

```
a.reshape((5,1))
```

![17](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/17.png)


注：参考补充自： 
[https://blog.csdn.net/koala_tree/article/details/78045596](https://blog.csdn.net/koala_tree/article/details/78045596)
