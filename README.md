# Data_Mining_Toolbox
使用机器学习、深度学习等方式进行数据挖掘时常用函数工具箱.

包含的工具函数库主要包括:  

1. #### common_helper.py

   各种数据挖掘任务都可能用到的工具。

   |               函数               |              描述              |
   | :------------------------------: | :----------------------------: |
   |  writebunchobj(path, bunchobj)   |     对象持久化为pickle文件     |
   |        readbunchobj(path)        |     读取持久化的pickle文件     |
   | nsigma_threehold(input_data,n=3) | 获取nSigma法进行异常检测的阈值 |

2. #### Plot_helper.py

   **画图**相关的工具函数。

   |                             函数                             |                   描述                   |
   | :----------------------------------------------------------: | :--------------------------------------: |
   | plot_curve(x,y,label,title,xlabel,ylabel,figsize=(8,6),ylim=None,grid=True,title_size=20,xylabel_size=15,legend_size=12) |                 画折线图                 |
   | plot_train_curve(epochs,train_loss_list,train_acc_list,val_loss_list=None,val_acc_list=None) | 画训练过程中的损失函数和准确率变化折线图 |
   |              plot_distribution(data,data_name)               |               画数据分布图               |

3. #### dl_helper.py

   **深度学习**工具函数。

   |                             函数                             |                             描述                             |
   | :----------------------------------------------------------: | :----------------------------------------------------------: |
   | train(model,train_x,train_y,val_x,val_y, epochs, batch_size,optimizer=None) | 模型训练，训练过程中每次在验证集上效果有提升时的参数进行保存，画模型效果变化图 |
   |           test(model, test_x, test_y, batch_size)            |                    在测试集上进行效果测试                    |
   |                predict(model, x, batch_size)                 |                      对指定数据进行预测                      |





其他常用工具：

1. #### torchsummaryM

   &emsp;&emsp;常用进行**模型各层参数个数、占用内存情况统计**的工具函数。在进行参数统计、内存占用情况时更加普遍被人熟知的工具是torchsummary，但是torchsummary存在对RNN模型不支持、接口奇葩等问题，很容易由于使用原因造成各种bug，因此更加推荐torchsummaryM，**是torchsummarry的进化版，该工具不仅支持RNN、而且接口正常、展示效果更好**。

   ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.sxatnmwkeie.png)

   &emsp;&emsp;**单个输入**:
   
   ~~~shell
   torchsummaryM.summary(model, torch.randint(0,50,(16,50)).to(device))
   ~~~
   
   &emsp;&emsp;**多输入**:
   
   ~~~shell
   torchsummaryM.summary(model, torch.randn((16,15)),torch.randn((16,2)))
   ~~~
   
   >  虽然torchsummary有各种缺陷，但如果非要尝试一下，看[这里](torchsummary.md)来脱坑。

