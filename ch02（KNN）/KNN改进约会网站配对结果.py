import numpy as np
import operator

'''K近邻算法'''
def classify0(inX,dataSet,labels,k):
    # 计算距离
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
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

'''解析文本'''
def file2matrix(filename):
    with open(filename,mode='r') as fr:
        arrayOfLines=fr.readlines()
        numberOfLines=len(arrayOfLines)
        returnMat=np.zeros((numberOfLines,3))
        classLabelVector=[]
        index=0
        for line in arrayOfLines:
            line=line.strip()
            listFromLine=line.split('\t')
            returnMat[index,:]=listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index+=1
        return returnMat,classLabelVector

dataX,labels=file2matrix('D:/PythonFile/Machine Learning in Action/ch02/datingTestSet2.txt')

'''数据分析'''
import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)
ax.scatter(dataX[:,0],dataX[:,2],c=15 *np.array(labels),s=15 *np.array(labels))
plt.show()

'''数据归一化'''
def autoNorm(dataX):
    # 归一化公式： newVal=(oldval-min)/(max-min)
    minVals=dataX.min(axis=0)
    maxVals=dataX.max(axis=0)
    ranges=maxVals-minVals
    rows=dataX.shape[0]
    newVal=dataX-np.tile(minVals,(rows,1))
    newVal=newVal/np.tile(ranges,(rows,1))
    return newVal,ranges,minVals

'''测试算法：验证分类器'''
def datingClassTest():
    hoRatio=0.1
    # 读入数据
    filename='D:/PythonFile/Machine Learning in Action/ch02/datingTestSet2.txt'
    dataX,labels=file2matrix(filename)
    # 归一化
    normMat,ranges,minVals=autoNorm(dataX)

    m=dataX.shape[0]
    numTestVecs=int(m*hoRatio)
    errorcount=0
    traingSet=normMat[numTestVecs:m]
    traingLabels=labels[numTestVecs:m]
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],traingSet,traingLabels,5)
        if(classifierResult!=labels[i]):
            errorcount=errorcount+1
    print("error rate :%f" % ((errorcount) / (numTestVecs))) #0.05

'''使用算法：构建完整可用系统'''
def classifyPerson():
    resultList = ["第一类", "第二类", "第三类"]  # output lables

    percentTats = float(input("玩游戏消耗的时间"))
    ffilm = float(input("每年获得的飞行里程数"))
    iceCream = float(input("每周消费冰淇淋"))
    # 读入数据
    filename='D:/PythonFile/Machine Learning in Action/ch02/datingTestSet2.txt'
    dataX, labels = file2matrix(filename)
    # 归一化
    normMat, ranges, minVals = autoNorm(dataX)

    test_list = np.array([percentTats, ffilm, iceCream])
    classifierResult = classify0(test_list, dataX, labels, 3)
    print("你喜欢的类别:" + resultList[classifierResult])
