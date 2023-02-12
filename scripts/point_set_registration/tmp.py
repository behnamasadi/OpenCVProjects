from cv2 import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math as math
#from scipy import spatial



import sys
print(sys.version)
import warnings
warnings.filterwarnings('ignore')

#from  sklearn import *
from sklearn.neighbors import KDTree


# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

#plt.show()




newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 5)


print( "result: ", results,"\n")
print( "neighbours: ", neighbours,"\n")
print( "distance: ", dist)

plt.show()







tmp=np.array([ [1, 1], [2, 2], [3, 3], [4, 4], [5, 5] ]    )
qurey_point=np.array(    [ [1.5, 1.4] ,  [1.6, 1.9] , [4.5, 4.9] ,[2, 3]  ]    )


print( "qurey_point.shape", qurey_point.shape)

tree = KDTree(tmp)
nearest_dist, nearest_ind = tree.query(qurey_point, k=1)  # k=2 nearest neighbors where k1 = identity
print(nearest_ind)     # drop id
print(tmp[nearest_ind])





result=tmp[nearest_ind].reshape(4,2)
print( result)


print( "--------------------------------------")
print( result[:,0])
print( "--------------------------------------")
print( result[:,1])


x1=result[:,0]
y1=result[:,1]

x2=qurey_point[:,0]
y2=qurey_point[:,1]



plt.scatter(tmp[:,0] , tmp[:,1] )
plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)

plt.show()

