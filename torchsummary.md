

## torchSummary

&emsp;&emsp;常用进行模型各层参数个数、占用内存情况进行统计的工具函数。

**参数**：

- model: 模型

- input_size: (深度,其他)，输入向量大小，去除batch_size

  > 注意:这里必须包含深度，不能直接一维

- batch_size=-1: 要测试的数据量

- device=device(type='cuda', index=0): 是否使用GPU以及指定GPU号

- dtypes=None, 更改生成的测试数据类型，**格式为：[[torch.LongTensor]*len(input_size)]**

~~~python
torchsummary.summary(model, (1,50),64)
~~~



### 常见的问题

1. #### 使用pip安装后没有dtypes参数

   - **问题分析**: 这个问题主要是因为作者只更新了github版本，pip源中的版本没有进行更新，导致pip安装的版本中没有dtypes参数

   - **解决方法**: 在github下载最新版本进行源码安装

2. #### 包含Embedding层的模型中出现”Expected tensor for argument #1 'indices' to have scalar type Long; but got CUDAType instead (while checking arguments for embedding)“

   - **问题分析**: 由于该工具函数中不仅进行dtype指定则默认生成FloatTensor进行测试，而Embedding层只能接收LongTensor类型的数据

     ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.vw2d9lsq6sn.png)

   - 解决方法: 在dtypes中进行指定输入数据类型

     ~~~python
     torchsummary.summary(model, (1,50),dtypes=[torch.LongTensor]*len((1,50)))
     ~~~


3. #### 输入数据本身为两维向量报错

   &emsp;&emsp;在torchsummary中input_size至少为两维，不能缺少深度。

   &emsp;&emsp;举个例子来说，在模型的输入向量为(batch_size,feature_size),那么如果使用torchsummaryM则为:

   ~~~
   torchsummaryM.summary(model, torch.randfn((16,50)).to(device))
   ~~~

   &emsp;&emsp;转化到torchsummary，只能为：

   ~~~
   torchsummary.summary(model, torch.randint((50,)).to(device))
   ~~~

   &emsp;&emsp;而不能是：

   ~~~
   torchsummary.summary(model, torch.randint((50)).to(device))
   ~~~

   &emsp;&emsp;将直接报错：

   ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.xd1ee6ec9oe.png)

