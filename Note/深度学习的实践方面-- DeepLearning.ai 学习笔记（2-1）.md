>**课程笔记地址**：https://mp.csdn.net/postlist
**课程代码地址**：https://github.com/duboya/DeepLearning.ai-pragramming-code/tree/master
欢迎大家**fork**及**star**！(-^O^-)



# 改善深层神经网络：超参数调试、正则化以及优化 —深度学习的实践方面
## 1. 训练、验证、测试集
对于一个需要解决的问题的样本数据，在建立模型的过程中，我们会将问题的data划分为以下几个部分：


- **训练集**（train set）：用训练集对算法或模型进行训练过程；
- **验证集**（development set）：利用验证集或者又称为简单交叉验证集（hold-out cross validation set）进行交叉验证，选择出最好的模型；
- **测试集**（test set）：最后利用测试集对模型进行测试，获取模型运行的无偏估计。
###小数据时代
在小数据量的时代，如：100、1000、10000的数据量大小，可以将data做以下划分：


- 无验证集的情况：70% / 30%；
- 有验证集的情况：60% / 20% / 20%；
通常在小数据量时代，以上比例的划分是非常合理的。
###大数据时代
但是在如今的大数据时代，对于一个问题，我们拥有的data的数量可能是百万级别的，所以验证集和测试集所占的比重会趋向于变得更小。


验证集的目的是为了验证不同的算法哪种更加有效，所以验证集只要足够大能够验证大约2-10种算法哪种更好就足够了，不需要使用20%的数据作为验证集。如百万数据中抽取1万的数据作为验证集就可以了。


测试集的主要目的是评估模型的效果，如在单个分类器中，往往在百万级别的数据中，我们选择其中1000条数据足以评估单个模型的效果。


- 100万数据量：98% / 1% / 1%；
- 超百万数据量：99.5% / 0.25% / 0.25%（或者99.5% / 0.4% / 0.1%）
###Notation
- 建议验证集要和训练集来自于同一个分布，可以使得机器学习算法变得更快；
- 如果不需要用无偏估计来评估模型的性能，则可以不需要测试集。
> 1. 训练集与测试集必须来自同一数据分布，不然测试集性能无法衡量模型偏差，后续Ng会介绍当
> 2. 对开发集和测试集上的数据进行检查，确保他们来自于相同的分布。使得我们以开发集为目标方向，更正确地将算法应用到测试集上。
> 3. "在深度学习的时代，因为需求的数据量非常大，现在很多的团队，使用的训练数据都是和开发集和测试集来自不同的分布。"
> 4. 后面课程中Ng也提到，如做一个鉴定猫狗图片的分类器，训练数据往往是高清图片，但用户上传的照片质量则普遍偏低，这时候，train set、dev set来自高清图片，测试集来自用户上传照片，势必造成分类效果很差。
> 5. 此时的做法通常有两种：一种是将收集到的少量实际数据如10000张与实际高清训练数据200000张照片打乱再依次分配到train set, dev set, test set。
> 虽然这种方式实现了数据的平均分布，dev set 与 test set也来自同一分布。但由于实际数据集占总数据集比例很小，此时，训练出的模型更倾向于高清照片分类，所以这种做法效果并不好。（不推荐）
> 6. 另一种做法是采用训练集全部采用高清照片，dev set与test set采用实际数据集，或者从实际数据集中分出5000张加入到训练数据集中。
> 此时，好处是：开发集全部来自手机图片，瞄准目标； 坏处则是：训练集和开发、测试集来自不同的分布。
从长期来看，这样的分布能够给我们带来更好的系统性能。（推荐)
> 7. 通过估计学习算法的偏差和方差，可以帮助我们确定接下来应该优先努力的方向。但是当我们的训练集和开发、测试集来自不同的分布时，分析偏差和方差的方式就有一定的不同。





## 2. 偏差、方差
对于下图中两个类别分类边界的分割： 

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.1.png)


从图中我们可以看出，在欠拟合（underfitting）的情况下，出现高偏差（high bias）的情况；在过拟合（overfitting）的情况下，出现高方差（high variance）的情况。


在bias-variance tradeoff 的角度来讲，我们利用训练集对模型进行训练就是为了使得模型在train集上使 bias 最小化，避免出现underfitting的情况；


但是如果模型设置的太复杂，虽然在train集上 bias 的值非常小，模型甚至可以将所有的数据点正确分类，但是当将训练好的模型应用在dev 集上的时候，却出现了较高的错误率。这是因为模型设置的太复杂则没有排除一些train集数据中的噪声，使得模型出现overfitting的情况，在dev 集上出现高 variance 的现象。


所以对于bias和variance的权衡问题，对于模型来说是一个十分重要的问题。

### 例子：
几种不同的情况： 


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.2.png)


以上为在人眼判别误差在0%的情况下，该最优误差通常也称为“贝叶斯误差”，如果“贝叶斯误差”大约为15%，那么图中第二种情况就是一种比较好的情况。
> 1. 上图中optimal (Bayes) error约为0，Bayes error是理论极限达到的最小错误，由于人非常擅长处理图像、音频之类的非结构化数据处理，其处理性能已逼近理论极限，故而常用人在这类事务上处理的error当做是理论上能达到的最小error，这也往往是train set训练模型力求达到的目标。
> 2. 如果训练集距离Bayes error差距较大，则证明模型没有训练好，存在high bias，如果train set error $\approx$ Bayes error，则证明不存在high bias，此时，再分析dev set error，若dev set error $\approx$ train set error，则证明不存在high variance, 若dev set error >> train set error，则证明存在过拟合。

**High bias and high variance的情况**


上图中第三种bias和variance的情况出现的可能如下：

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.3.png)

>即训练的模型既存在高偏差(high bias)，又存在高方差(high variance)。这种情况在高维空间更常见：在高维空间中更容易存在部分空间过拟合，部分空间欠拟合现象。

## 3. 机器学习的基本方法
在训练机器学习模型的过程中，解决High bias 和High variance 的过程：

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.4.png)


- 1.是否存在High bias ? 
	- 增加网络结构，如增加隐藏层数目；
	- 训练更长时间；
	- 寻找合适的网络架构，使用更大的NN结构；
- 2.是否存在High variance？ 
	- 获取更多的数据；
	- 正则化（ regularization）；
	- 寻找合适的网络结构；


在大数据时代，深度学习对监督式学习大有裨益，使得我们不用像以前一样太过关注如何平衡偏差和方差的权衡问题，通过以上方法可以使得在不增加另一方的情况下减少一方的值。
> 1. 机器学习中variance 与 bias 往往存在一个权衡取舍的问题，要么增大bias，来减少variance（如logistic regression减少输入变量），要么增大variance，来减少bias(如random forest增大tree的数量)。
> 2. 而neural networks往往不需这样的权衡，可在不增大bias的情况下减少variance，同理也可在不增大variance的情况下减少bias。

## 4. 正则化（regularization）


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.5.png)


## 5. 为什么正则化可以减小过拟合
假设下图的神经网络结构属于过拟合状态： 

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.6.png)


对于神经网络的Cost function：

$$
J(w^{[1]}, b^{[l]}, \cdots , w^{[L]}, b^{[L]}) = \frac{1}{m}\sum_{i=1}^{m}l(\hat{y}^{(i)},y^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L}\Arrowvert w^{[l]}\Arrowvert_{F}^{2}
$$


加入正则化项，直观上理解，正则化因子$\lambda$设置的足够大的情况下，为了使代价函数最小化，权重矩阵W就会被设置为接近于0的值。则相当于消除了很多神经元的影响，那么图中的大的神经网络就会变成一个较小的网络。

当然上面这种解释是一种直观上的理解，但是实际上隐藏层的神经元依然存在，但是他们的影响变小了，便不会导致过拟合。


数学解释：

假设神经元中使用的激活函数为$g(z) = tanh⁡(z)$，在加入正则化项后： 

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.7.png)


当$\lambda$增大，导致$W^{[l]}$减小，$Z^{[l]} = W^{[l]}a^{[l−1]}+b^{[l]}$便会减小，由上图可知，在z较小的区域里，$tanh⁡(z)$函数近似线性，所以每层的函数就近似线性函数，整个网络就成为一个简单的近似线性的网络，从而不会发生过拟合。

> 注：由以上分析也应当得知，lambda应该设定合理，不然lambda过大的话，整个neural network变成了线型函数的叠加，依旧是线型函数，模型表达能力大大降低。


## 6. Dropout 正则化
Dropout（随机失活）就是在神经网络的Dropout层，为每个神经元结点设置一个随机消除的概率，对于保留下来的神经元，我们得到一个节点较少，规模较小的网络进行训练。

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.8.png)


### 实现Dropout的方法：反向随机失活（Inverted dropout）
首先假设对 layer 3 进行dropout：

```python
keep_prob = 0.8  # 设置神经元保留概率
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 /= keep_prob
```

这里解释下为什么要有最后一步：a3 /= keep_prob

依照例子中的keep_prob = 0.8 ，那么就有大约20%的神经元被删除了，也就是说$a^{[3]}​$中有20%的元素被归零了，在下一层的计算中有$Z^{[4]} = W^{[4]}a^{[3]} + b^{[4]}​$，所以为了不影响$Z^{[4]}​$的期望值，所以需要$W^{[4]}⋅a^{[3]}​$的部分除以一个keep_prob。

Inverted dropout通过对“a3 /= keep_prob”,则保证无论keep_prob设置为多少，都不会对$Z^{[4]}$的期望值产生影响。

Notation：在测试阶段不要用dropout，因为那样会使得预测结果变得随机。

> 1. dropout主要用于CV方向，由于CV方向input size很大，输入了太多像素，以至于没有足够多的数据，所以一直存在过拟合，故而常用到dropout，几乎成了默认设置！
> 2. 但dropout是一种正则化手段，除非算法表现出过拟合，不然不用使用dropout，故而dropout在其他方向应用很少。因为即便是模型表现出了过拟合，也有很多方法可以用来对抗过拟合（比如使用L2正则式（很常用），使用L1正则式，加入更多数据，更改网络机构，提前结束训练）等方法。
3. 一般输入层很少用到dropout，即对于输入层常设置keep_prob =1；
4. drop out实施时候，可采用不同方式，一种方式是针对不同层设置不同的keep_prob，对应层神经元数目过多的时候，设置keep_prob较小（如0.5-0.8），对应层神经元数目过少的时候，设置keep_porb较大，如设置0.8，0.9等，这时候每层设置的keep_prob也是一个超参数，需要使用交叉验证寻找超参，增加了训练难度；
另一种方法只针对神经元数目较多的层设置相同的drop_prob，这时候，只增加了一个keep_prob超参。
5. 因为引入dropout之后程序，cost function难以明确定义，程序变得难以调试，故而Ng的通常做法是先关闭dropout，设置keep_prob=1，运行代码，保证损失函数J单调递减，然后再打开dropout函数，希望在dropout过程中，代码并未引入bug。



## 7. 理解 Dropout
另外一种对于Dropout的理解。

这里我们以单个神经元入手，单个神经元的工作就是接收输入，并产生一些有意义的输出，但是加入了Dropout以后，输入的特征都是有可能会被随机清除的，所以该神经元不会再特别依赖于任何一个输入特征，也就是说不会给任何一个输入设置太大的权重。

所以通过传播过程，dropout将产生和L2范数相同的**收缩权重**的效果。

对于不同的层，设置的**keep_prob**也不同，一般来说神经元较少的层，会设keep_prob =1.0，神经元多的层，则会将keep_prob设置的较小。

**缺点：**

dropout的一大缺点就是其使得 Cost function不能再被明确的定义，以为每次迭代都会随机消除一些神经元结点，所以我们无法绘制出每次迭代$J(W,b)$下降的图，如下：

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.9.png)


**使用Dropout：**

- 关闭dropout功能，即设置 keep_prob = 1.0；
- 运行代码，确保$J(W，b)$函数单调递减；
- 再打开dropout函数。

## 8. 其他正则化方法
- 数据扩增（Data augmentation）：通过图片的一些变换，得到更多的训练集和验证集； 


> Data augmentation是一种常用方法，会在第三课中详细讲述，大致有对称变换等方式。

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.10.png)

- Early stopping：在交叉验证集的误差上升之前的点停止迭代，避免过拟合。这种方法的缺点是无法同时解决bias和variance之间的最优。 

> 这种方法Ng并不推荐用，因为按照Ng在第三课中讲到的正交性原则，设计、训练模型的时候应该使调整bias与调整variance的方法分开，互不影响，这样在模型出现hig bias or high variance的时候就可以针对问题进行单独处理而不会影响另一方。

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.11.png)




## 9. 归一化输入
对数据集特征$x_1$, $x_2$归一化的过程： 

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.12.png)


- 计算每个特征所有样本数据的均值：$\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}$
- 减去均值得到对称的分布：$x := x − \mu$；
- 归一化方差：$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}x^{(i)^2}$, $x = x / \sigma^2$

> 这是一种高斯归一化方法。
###使用归一化的原因：


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.13.png)


由图可以看出不使用归一化和使用归一化前后Cost function 的函数形状会有很大的区别。

在不使用归一化的代价函数中，如果我们设置一个较小的学习率，那么很可能我们需要很多次迭代才能到达代价函数全局最优解；如果使用了归一化，那么无论从哪个位置开始迭代，我们都能以相对很少的迭代次数找到全局最优解。


## 10. 梯度消失与梯度爆炸
如下图所示的神经网络结构，以两个输入为例：

![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.14.png)


上面的情况对于导数也是同样的道理，所以在计算梯度时，根据情况的不同，梯度函数会以指数级递增或者递减，导致训练导数难度上升，梯度下降算法的步长会变得非常非常小，需要训练的时间将会非常长。


在梯度函数上出现的以指数级递增或者递减的情况就分别称为梯度爆炸或者梯度消失。


> 1. 梯度消失带来的问题是梯度无法有效回传，当从最后一层算出loss后，最后几层还能进行梯度下降，但越往前回传，梯度改变量越小，还没到中间就接近于0，造成前面的层无法得到训练，其最终结果就是虽然层数很多，但是前面层得不到训练，模型最终依旧是表现为浅层模型（只有最后几层起到作用）。
> 2. 而梯度爆炸则直接使得前面层变化太大，导致参数数值溢出，最终前面层参数表现为Nan。
> 3. 其实梯度爆炸和梯度消失问题都是因为网络太深，网络权值更新不稳定造成的，本质上是因为梯度反向传播中的连乘效应。对于更普遍的梯度消失问题，可以考虑用Relu激活函数取代sigmoid函数。另外，LSTM的结构设计也可以改善RNN的梯度消失问题。


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.19.png)



> 1. 梯度爆炸比梯度消失更容易解决，也更容易判定，如果出现梯度爆炸，其经过BP参数更新，w会出现指数级增长（如上分析），导致最终w数值溢出，会造成前层神经网络出现很多Nan，这时候便可以判定是否出现梯度爆炸。
> 2. 解决办法就是利用gradient clipping，对w设置一个上线，当达到这个上限之后就对其进行缩放，保证w不至于太大。
> 3. 做gradient clipping有很多方法，在RNN编程实践的时候提到一种简单的方法，即设置上下线[-N, +N]，当达到这个上下线的时候就用上下线阈值替代w。对于梯度消失问题 ，在RNN结构中是我们首要关心的问题，也更难解决。
> 4. 对于梯度消失问题，在RNN的结构中是我们首要关心的问题，也更难解决；虽然梯度爆炸在RNN中也会出现，但对于梯度爆炸问题，因为参数会指数级上升，会让我们的网络参数变得很大，得到很多的Nan或者数值，所以梯度爆炸是很容易发现的，我们的解决方法就是用梯度修剪，也就是观察梯度变量，如果其大于某个阈值，则对其进行缩放，保证它不会太大。


## 11. 利用初始化缓解梯度消失和爆炸问题
以一个单个神经元为例子： 


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.15.png)


由上图可知，当输入的数量n较大时，我们希望每个$w_i$的值都小一些，这样它们的和得到的z也较小。

这里为了得到较小的$w_i$，设置$Var(w_i) = \frac{1}{n}$，这里称为Xavier initialization。 
对参数进行初始化：


```
WL = np.random.randn(WL.shape[0],WL.shape[1])* np.sqrt(1/n)
```


这么做是因为，如果激活函数的输入x近似设置成均值为0，标准方差1的情况，输出z也会调整到相似的范围内。虽然没有解决梯度消失和爆炸的问题，但其在一定程度上确实减缓了梯度消失和爆炸的速度。

**不同激活函数的 Xavier initialization：**

- 激活函数使用Relu：$Var(w_i) = \frac{2}{n}$
- 激活函数使用tanh：$Var(w_i) = \frac{1}{n}$
其中n是输入的神经元个数，也就是$n^{[l−1]}$。


## 12. 梯度的数值逼近
使用双边误差的方法去逼近导数： 


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.16.png)


由图可以看出，双边误差逼近的误差是0.0001，先比单边逼近的误差0.03，其精度要高了很多。

涉及的公式：


- 双边导数：

$$
f^{'}(\theta) = \lim_{\epsilon \rightarrow 0} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}
$$


误差：$O(\epsilon^2)$

- 单边导数：

$$
f^{'}(\theta) = \lim_{\epsilon \rightarrow 0} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{\epsilon}
$$


误差：$O(\epsilon)$


## 13. 梯度检验
下面用前面一节的方法来进行梯度检验。

### 连接参数


因为我们的神经网络中含有大量的参数：$W^{[1]}$, $b^{[1]}$, $\cdots$, $W^{[L]}$, $b^{[L]}$，为了做梯度检验，需要将这些参数全部连接起来，reshape成一个大的向量$\theta$。

同时对$dW^{[1]}$, $db^{[1]}$, $\cdots$, $dW^{[L]}$, $db^{[L]}$执行同样的操作。 


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.17.png)


### 进行梯度检验
进行如下图的梯度检验： 


![](https://raw.githubusercontent.com/duboya/DeepLearning.ai-pragramming-code/master/Note_image/21.18.png)


判断$d\theta_{approx} \approx d\theta$是否接近。

判断公式： 
$$
\frac{\parallel d\theta_{approx} - d\theta \parallel_2}{\parallel d\theta_{approx}\parallel_2 + \parallel d\theta \parallel_2}
$$


其中，"$\parallel \cdot \parallel_2$"表示欧几里得范数，它是误差平方之和，然后求平方根，得到的欧氏距离。


## 14. 实现梯度检验 Notes
- 不要在训练过程中使用梯度检验，只在debug的时候使用，使用完毕关闭梯度检验的功能；
- 如果算法的梯度检验出现了错误，要检查每一项，找出错误，也就是说要找出哪个$d\theta_{approx}[i]$与$dθ$的值相差比较大；
- 不要忘记了正则化项；
- 梯度检验不能与dropout同时使用。因为每次迭代的过程中，dropout会随机消除隐层单元的不同神经元，这时是难以计算dropout在梯度下降上的代价函数J；
- 在随机初始化的时候运行梯度检验，或许在训练几次后再进行。

>注：补充参考自： 
> Data augmentation是一种常用方法，会在第三课中详细讲述，大致有对称变换等方式。
