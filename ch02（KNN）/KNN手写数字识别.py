import numpy as np
import matplotlib.pyplot as plt
import operator
import os

'''将图像转换为测试向量'''
def img2vector(filename):
    returnVect=np.zeros((1,1024))
    with open(filename,mode='r') as fr:
        lineStr=fr.readlines()
        for i in range(32):
            for j in range(32):
                returnVect[0,i*32+j]=lineStr[i][j]
    return returnVect

filename='D:/PythonFile/Machine Learning in Action/ch02/digits/trainingDigits/0_13.txt'
testvector=img2vector(filename)

'''k近邻算法'''
def classify0(inX,dataX,labels,k):
    datasize=dataX.shape[0]
    diffMat=np.tile(inX,(datasize,1))-dataX
    sqDiffMat=diffMat**2
    sqDistence=sqDiffMat.sum(axis=1)
    distance=sqDistence**0.5
    sortedDistanceIndicies=distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistanceIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),
                            reverse=True)
    return sortedClassCount[0][0]

'''测试代码'''
# 主要是对文件和label对应的处理
def handwritingClassTest():
    # 提取trainingSet
    trainingFileList=os.listdir('D:/PythonFile/Machine Learning in Action/ch02/digits/trainingDigits')
    m=len(trainingFileList)
    traingMat=np.zeros((m,1024))
    hwLabels = []  # 每个文件对应数字（即标签）
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split(".")[0]
        classNumStr=fileStr.split("_")[0]
        hwLabels.append(classNumStr)
        traingMat[i,:]=img2vector("D:/PythonFile/Machine Learning in Action/ch02/digits/trainingDigits/%s" % fileNameStr)

    testFileList=os.listdir('D:/PythonFile/Machine Learning in Action/ch02/digits/trainingDigits')
    mtest=len(testFileList)
    errorcount=0
    for i in range(mtest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split(".")[0]
        classNumStr=fileStr.split("_")[0]
        vectorUderTest=img2vector("D:/PythonFile/Machine Learning in Action/ch02/digits/trainingDigits/%s" % fileNameStr)
        classifierResult=classify0(vectorUderTest,traingMat,hwLabels,3)
        if classifierResult!=classNumStr:
            errorcount=errorcount+1.0
    print("errorcount:%d" %errorcount)
    print("error rate : %f" % float((errorcount)/mtest))

handwritingClassTest()
# errorcount:23
# error rate : 0.011892