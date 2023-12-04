import numpy as np

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


def rot2(theta):
    R = np.zeros([2, 2])

    R = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
    return R

# [
#  [-1] ,
#  [-2]
# ]


f = np.random.random([4, 1])

# x[0:2, 0:1] = np.array([[-1], [-2]])
# print(x[0:2, 0:1])
# print(x@f)


# edge = [[float(6), float(7), float(8)],
#         [float(7), float(
#             9), float(10)],
#         [float(8), float(10), float(11)]]

# print(np.array(edge))
# print(np.array(edge).T)


# H = np.zeros([5, 5])


# H[0:3, 0:3] = H[0:3, 0:3] + 1e6*np.eye(3)

# print(H)

# x = np.random.random([3, 1])
# print(x)
# print(np.linalg.norm(x))


x = [1, 2, 3]
for index, value in enumerate(x):
    x[index] = x[index] * 2
    print(index)

print(x)

# theta1 = np.pi/2

# R_1toG = rot2(theta1)
# print(R_1toG)

# # y = R_1toG@np.array([[2], [3]])
# a, b = R_1toG@[[2], [3]]

# print(a, b)
