
import cv2
import numpy as np
import sys
from numpy.random import *
import math as math


import matplotlib.pyplot as plt


data1 = np.array([[0.25, 0.25], [0.5, 0.25], [0.5, 0.5], [0.25, 0.5]])
print(data1.shape)
#data1=np.array([ [1, 2] , [3, 4 ], [5, 6 ] , [7, 8 ] ] )
# data1=data1.reshape(2,4)


#data1_src = np.array([data1.T], copy=True).astype(np.float32)


# print data1_src.shape


# tetha=math.pi/4
tetha = 0
tx = 0
ty = 0
Tr = np.array([[np.cos(tetha), -np.sin(tetha), tx],
               [np.sin(tetha), np.cos(tetha),  ty],
               [0,         0,          1]])


Tr = np.array([[np.cos(tetha), -np.sin(tetha), tx],
               [np.sin(tetha), np.cos(tetha),  ty]])

print(Tr.shape)


dst = np.array([[0.25, 0.25], [0.5, 0.25], [0.5, 0.5], [0.25, 0.5]])
w = 4
h = 2
dst = cv2.warpAffine(data1, Tr, (w, h))


xmin = -2
xmax = 2
ymin = -2
ymax = 2

plt.axis([xmin, xmax, ymin, ymax])


print(data1[:, 0])
print(data1[:, 1])

print(dst)
print(dst.shape)

print(dst[:, 0])
print(dst[:, 1])


#
# dst=dst.reshape(4,2)
#
# #plt.scatter( data1[:,0] , data1[:,1]  )
# plt.scatter( dst[:,0] , dst[:,1]  )
# plt.show()
#
#
#
#
#
# pts1 = np.float32([[0,0],[0,10],[10,10]])
# pts2 = np.float32([[5,0],[5,10],[15,10]])
# M = cv2.getAffineTransform(pts1,pts2)
#
# print "-------------------------------------------------------"
# print M
# # print pts1.shape([])
#
# rows=2
# cols=3
# pts1 = np.array([[0,0],[0,10],[10,10]], dtype=np.float32)
# print 'type pts1', type(pts1)
# print 'pts1 shape', pts1.shape, pts1.dtype
# image=cv2.imread('lena.jpeg',cv2.IMREAD_GRAYSCALE)
# print 'lena shape',image.shape
# print image.shape
# rows, cols = image.shape[:2]
# print rows, cols
#
# image=pts1
# #rows=3
# #cols=2
# #print image
# image = cv2.warpAffine(src=image, M=M, dsize=(10,10))
# print image
# #print image
# cv2.imshow('image',image)
# cv2.waitKey(0)
#
# #image = cv2.warpAffine(pts1, M, (cols, rows))
# #print image


src = np.ones((100, 100), dtype=np.float32)
dst = np.ones((100, 100), dtype=np.float32)


p1x1 = 10
p1y1 = 10

p1x2 = 10
p1y2 = 20

p1x3 = 20
p1y3 = 10


src[p1x1, p1y1] = 0
src[p1x2, p1y2] = 0
src[p1x3, p1y3] = 0

p2x1 = p1x1
p2y1 = p1y1+20

p2x2 = p1x2
p2y2 = p1y2+20

p2x3 = p1x3
p2y3 = p1y3+20


src[p2x1, p2y1] = 0
src[p2x2, p2y2] = 0
src[p2x3, p2y3] = 0

pts1 = np.float32([[0, 0], [0, 10], [10, 10]])


tetha = 0
tx = 0
ty = 20

Tr = np.array([[np.cos(tetha), -np.sin(tetha), tx],
               [np.sin(tetha), np.cos(tetha),  ty]])


print('-----------------------------------')

pts1 = np.float32([[p1x1, p1y1], [p1x2, p1y2], [p1x3, p1y3]])
pts2 = np.float32([[p2x1, p1y1], [p2x2, p1y2], [p2x3, p1y3]])


dst = cv2.warpAffine(src, Tr, src.shape)

print(pts1.shape)
print(pts2.shape)

M = cv2.getAffineTransform(pts1, pts2)
#M_estimate = cv2.estimateRigidTransform(pts1, pts2)
M_estimate = cv2.estimateAffine2D(pts1, pts2)

# print np.arccos(M_estimate[0][0]) / 2 / np.pi * 360


print(M)
# cv2.imshow("src",src)
cv2.imshow("dst", dst)
cv2.waitKey(0)


print("********************")
print(type(pts1))
print(pts1.shape)
# print pts1[0]
# print pts1[1]
# print pts1[2]

print(pts1[:0])
print("********************")

print(pts1[:1])
print("********************")

print(pts1[:2])


# print pts1[1:]
# print pts1[0]
# print pts1[1]
# print pts1[2]
#
# plt.scatter(pts1[:0], pts1[:1] )
# plt.scatter(pts2[:0], pts2[:1] )
#
# plt.show()
