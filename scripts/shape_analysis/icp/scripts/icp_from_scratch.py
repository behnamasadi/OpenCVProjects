import matplotlib.pyplot as plt
import numpy as np
import math as math
#from scipy import spatial



import sys
print(sys.version)
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KDTree

def SVDTransformFinder(X,P):
    
    
    Mu_X=np.mean(X, axis=1,keepdims=True)
    Mu_P=np.mean(P, axis=1,keepdims=True)
    
    X_Prime=X-Mu_X
    P_Prime=P-Mu_P
    W=np.dot(X_Prime,P_Prime.T)
    
    u,s,vh=np.linalg.svd(W , full_matrices=True)
    
    
    print("---------------SVD result----------------\n") 

    R=np.matmul(u,vh)
    print ("rotation: ",R)
    T=Mu_X-np.matmul(R,Mu_P)
    print ("Translation: ",T)
    return R,T
    


if __name__ == "__main__":
    
    point_count = 50
    th = np.pi / 8
    move = np.array([[0.40], [0.5]])
    x1 = np.linspace(0, 1.1, point_count)
    y1 = np.sin(x1 * np.pi)
    d1 = np.array([x1, y1])

    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d2 = np.dot(rot, d1) + move
    #d2 = np.dot(rot, d1) 
    #d2 = d1 + move
    
    print ("-------------original values---------------\n")
    print ("rot", rot)
    print ("move", move)
    
    print ("---------------------------------\n")

    
    
    xmin=-2
    xmax=6
    ymin=-1
    ymax=4

    plt.axis([xmin,xmax,ymin,ymax])
    
    
    
#Ploting points    
    plt.plot(d1[0], d1[1])
    plt.plot(d2[0], d2[1])
    
    x1=d1[0]
    y1=d1[1]
    x2=d2[0]
    y2=d2[1]
    
    print ("---------------------------x1.shape-----------------------------\n")

    print (x1.shape)
    
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)

    
    
    R,T=SVDTransformFinder(d2,d1)
    
    
    d = np.dot(R, d1)+T
    
    
    print ("rotation: ",R)
    
    print ("Translation: ",T)
    
    plt.plot(d[0], d[1],color='green')

    plt.show()
    
    
    print ("-------------------------------------------------------------\n")




    

    
    
    tmp= d1.T
    qurey_point= d2.T 


  
    
    print ("qurey_point.shape", qurey_point.shape)
    tree = KDTree(tmp)
    nearest_dist, nearest_ind = tree.query(qurey_point, k=1)  # k=2 nearest neighbors where k1 = identity
    print ("nearest_ind", nearest_ind )    # drop id)
    print ("tmp[nearest_ind]", tmp[nearest_ind])

    result=tmp[nearest_ind]
    result=result.reshape(50,2)
    print ( "result.shape", result.shape)
    
     

     
 
    x1=result[:,0]
    y1=result[:,1]

    
     
    x2=qurey_point[:,0]
    y2=qurey_point[:,1]
    
    plt.axis([xmin,xmax,ymin,ymax])

    print ("--------------------------------------")    
    print (x2)
    print ("--------------------------------------")    
    print (y2)

    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)
    plt.scatter([x1, x2], [y1, y2])
    
    
    
    plt.show()







    
#     tmp=np.array([ [1, 1], [2, 2], [3, 3], [4, 4], [5, 5] ]    )
#     qurey_point=np.array(    [ [1.5, 1.4] ,  [1.5, 1.9]  ]    )
#     tree = KDTree(tmp)
#     nearest_dist, nearest_ind = tree.query(qurey_point, k=1)  # k=2 nearest neighbors where k1 = identity
#     print(nearest_ind)     # drop id
#     print(tmp[nearest_ind])

#     tree = KDTree(d1)
#     nearest_dist, nearest_ind = tree.query(d2, k=1)
#     
#     
#     
#     print "----------------------------nearest_ind---------------------------------\n"
# 
#     print nearest_ind
#     
#     print nearest_dist
# 
#     print "----------------------------########################---------------------------------\n"
# 
# 
# 
#     x1=d1[nearest_ind][0]
#     y1=d1[nearest_ind][1]
#     
#     
#     x1=x1.T
#     y1=y1.T
#     
#     print x1.shape
#     print y1.shape
#     
# 
# 
#     x1=x1.reshape(50,)
#     y1=y1.reshape(50,)
# 
#     
#     x2=d2[0]
#     y2=d2[1]
#     
#     
# 
#     
#     
# #    x2=x2.reshape(50,1)
# #    y2=y2.reshape(50,1)
# 
#     print x2.shape
#     print y2.shape
#     
#     plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)
# 
#     plt.show()
    
    