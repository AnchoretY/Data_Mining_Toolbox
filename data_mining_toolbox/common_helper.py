#========================================
#          使用pickle进行读写
#========================================

def writebunchobj(path, bunchobj):
    """
        将对象进行持久化，常用于将比较耗时的操作存储执行结果，下次直接进行调用即可，例如：向量化特征提取
    """
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

def readbunchobj(path):
    """
        读取持久化的pickle文件，和writebunchobj对应
    """
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

#========================================
#              异常检测
#========================================

def nsigma_threehold(input_data,n=3):
    """
        获取3Sigma法进行异常检测的阈值
        Parameters:
        -----------------
            input_data: 输入数据，series、list
            n: n sigma中的n,默认为3，int
        Return:
        -----------------
            lower_threehold: 正常数据下阈值
            upper_threehold: 正常数据上阈值
    """
    mean = input_data.mean()
    std = input_data.std()
    lower_threehold1 = mean-n*std
    upper_threehold = mean+n*std
    print("normal range:{} ~ {}".format(lower_threehold1,upper_threehold))
    return lower_threehold,upper_threehold
