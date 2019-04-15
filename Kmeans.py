'''首先需要明白的一点是输入数据变换到一维。
因为我们是对整个图像进行聚类，所以他们的灰度值都属于一个特征（维度）内的，
而图像属于二维的，所以不能直接当data输入进去，
需要将图像转化为一个长条或者长链的一维数据。我们说data结束数据，
每一个特征放一列，灰度图像聚类的灰度值就是一个特征。'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# 以灰色导入图像
img = cv2.imread('E:/OP/HT.jpg', 0)  # 读取图片 以灰度
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('original')
plt.xticks([]), plt.yticks([])

#####改变图像的维度
img1 = img.reshape((img.shape[0]*img.shape[1],1))
img1 = np.float32(img1)    ###float32 类型数据处理最快

##设定一个criteria(迭代模式的停止选择)  三个元素的元组（type，max_iter,epsilon）
####  cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止
####  cv2.TERM_CRITERIA_MAX_ITER ：迭代次数超过max_iter停止
####  两者合体 任一满足停止
criteria =(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#设定初始类中心 flag
###两种选择方式 cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

###应用K-means   kmeans(data, K, bestLabels, criteria, attempts, flags)
# bestLabels 预设分类标签 无的话设为None
# attempts：重复试验k-means算法次数，将会返回最好的一次结果
compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,5,flags)
compactness_1,labels_1,centers_1 = cv2.kmeans(img1,2,None,criteria,10,flags)
compactness_2,labels_2,centers_2 = cv2.kmeans(img1,2,None,criteria,15,flags)

#####1D转为2D
img2 = labels.reshape((img.shape[0],img.shape[1]))
img3 = labels_1.reshape((img.shape[0],img.shape[1]))
img4 = labels_2.reshape((img.shape[0],img.shape[1]))
####显示处理后的图片
plt.subplot(222),plt.imshow(img2,'gray'),plt.title('kmeans_attempts_5')
plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(img3,'gray'),plt.title('kmeans_attempts_10')
plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(img4,'gray'),plt.title('kmeans_attempts_15')
plt.xticks([]),plt.yticks([])
plt.savefig("kmeans_attempts.png")
plt.show()





