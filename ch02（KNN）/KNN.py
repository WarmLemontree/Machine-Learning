from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
group,labels=createDataSet()

'''K近邻算法'''
def classify0(inX,dataSet,labels,k):
    # 计算距离
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distances=sqDistance**0.5
    # 对距离排序，返回排序后的下标
    sortedDistIndicies=distances.argsort()
    # 计算k个label出现的频率
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # 选出出现频率最高的label
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),
                            reverse=True) #从大到小排序
    return sortedClassCount[0][0]

classify0([0,0],group,labels,3) # B
