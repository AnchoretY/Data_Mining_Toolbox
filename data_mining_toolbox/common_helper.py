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
