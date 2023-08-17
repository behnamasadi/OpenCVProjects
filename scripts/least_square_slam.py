# export PATH="/home/$USER/usr/bin:$PATH"
# export LD_LIBRARY_PATH="/home/$USER/usr/lib:$LD_LIBRARY_PATH"

import g2o_file_reader
from load_2d_g2o import load_2d_g2o
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys

np.set_printoptions(threshold=sys.maxsize)


def getNode(nodes, id):
    for node in nodes:
        # print(node["id"])
        # print(id)
        if node["id"] == id:
            return node


def wrap2pi(theta):
    while theta.all() > math.pi:
        theta = theta - 2*math.pi

    while theta.all() < -math.pi:
        theta = theta + 2*math.pi
    return theta


def rot2(theta):
    R = np.zeros(2, 2)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
    return R


g2o_file = os.path.abspath("./data/slam/input_INTEL_g2o.g2o")
nodes, edges = load_2d_g2o(filename=g2o_file)
# print(nodes[1]["id"])
# print(nodes[1]["state"])
# # print(nodes)
# print(type(nodes))
# print(type(nodes[1]["state"]))
# print(type(nodes["state"]))


x_cords = [node["state"][0] for node in nodes]
y_cords = [node["state"][1] for node in nodes]
theta = [node["state"][2] for node in nodes]

plt.plot(x_cords, y_cords)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Nodes from g2o file')
plt.grid(True)
# plt.show()

# ensure our orientations are bounded between -pi and pi

theta = np.array(theta)

wrap2pi(theta)
# print(theta)

# for t in theta:
#     print(t)

# print(np.array(theta) / math.pi)

dx_norm = np.inf
iteration = 0
# print(len(x_cords))

print("edges[0]:", edges[0])
# print(nodes[0])

# {'id1': 0, 'id2': 1, 'meas': [0.0, 0.0, -0.000642], 'info': [[11.111271, -0.249667, 0.0], [-0.249667, 399.99984, 0.0], [0.0, 0.0, 2496.793089]]}

for edge in edges:

    # print(id1)
    # print(getNode(nodes, id1))
    node1 = getNode(nodes, id1)
    node2 = getNode(nodes, id2)

    x1 = node1["state"][0]
    y1 = node1["state"][1]
    theta1 = node1["state"][2]

    x2 = node2["state"][0]
    y2 = node2["state"][1]
    theta2 = node2["state"][2]

    theta2_in_1 = theta2-theta1

    # orientation error
    err_theta = wrap2pi(
        edge["meas"][3] - wrap2pi(node2["state"][3]-node1["state"][3]))

    # position error
    R_1toG = rot2(node1["state"][3])
    p_1inG = node1["state"][1:2]
    p_2inG = node2["state"][1:2]
    p_2in1 = np.matmul(np.transpose(R_1toG), (p_2inG - p_1inG))
    p_2in1_from_measurement = edge["meas"][1:2]
    err_pos = p_2in1_from_measurement - p_2in1

    # Jacobian of current relative in respect to NODE 1

    A_i_j = np.zeros(3, 3)
    A_i_j = [[-np.cos(theta1), -np.sin(theta1), -np.sin(theta1)(x2-x1) + np.cos(theta1)(y2-y1)],
             [np.sin(theta1), -np.cos(theta1), -np.cos(theta1)
              (x2-x1) - np.sin(theta1)(y2-y1)],
             [0, 0, -1]]

    # Jacobian of current relative in respect to NODE 2

    B_i_j = np.zeros(3, 3)
    B_i_j = [[np.cos(theta1), -np.sin(theta2), 0],
             [-np.sin(theta1), np.cos(theta1), 0], [0, 0, 1]]

    # update our information
    id1 = edge["id1"]
    id2 = edge["id2"]
    # H(3*id1+1:3*id1+3,3*id1+1:3*id1+3) = H(3*id1+1:3*id1+3,3*id1+1:3*id1+3) + Aij'*edge.info*Aij;
    # H(3*id1+1:3*id1+3,3*id2+1:3*id2+3) = H(3*id1+1:3*id1+3,3*id2+1:3*id2+3) + Aij'*edge.info*Bij;
    # H(3*id2+1:3*id2+3,3*id1+1:3*id1+3) = H(3*id2+1:3*id2+3,3*id1+1:3*id1+3) + Bij'*edge.info*Aij;
    # H(3*id2+1:3*id2+3,3*id2+1:3*id2+3) = H(3*id2+1:3*id2+3,3*id2+1:3*id2+3) + Bij'*edge.info*Bij;

    # update our error terms
    # b(3*id1+1:3*id1+3,1) = b(3*id1+1:3*id1+3,1) + Aij'*edge.info*[err_pos; err_theta];
    # b(3*id2+1:3*id2+3,1) = b(3*id2+1:3*id2+3,1) + Bij'*edge.info*[err_pos; err_theta];


# while dx_norm > 1e-2:
#     #our matrices for Hx=-b
#     b = np.zeros(3*len(x_cords), 1)
#     H = np.zeros(3*len(x_cords), 3*len(x_cords))
#     pass
