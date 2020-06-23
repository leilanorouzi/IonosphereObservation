
import numpy as np
# import matplotlib.pyplot as plt

# from sympy import *
import math
import cmath


def asCartesian(rthetaphi):
    # takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1]
    phi = rthetaphi[2]
    x = r * cmath.sin(theta) * cmath.cos(phi)
    y = r * cmath.sin(theta) * cmath.sin(phi)
    z = r * cmath.cos(theta)
    return [x, y, z]


def asSpherical(xyz):
    # print('asSpherical\n:',np.shape(xyz))
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = cmath.sqrt(np.dot(xyz, xyz))
    theta = cmath.acos(z / r)
    # phi     =  np.arctan2(y,x)
    phi = cmath.atan(y / x)
    return [r, theta, phi]


def rotate_vec(a, b):
    ref_p = asSpherical(a)
    # print(a)
    # print(ref_p)
    # print(type(ref_p[2]))
    # print([i * 180 / np.pi for i in ref_p[1:]])
    #
    # print(b)
    vec_p = asSpherical(b)
    # print(vec_p)
    # print([i * 180 / np.pi for i in vec_p[1:]])

    rotated = np.subtract(ref_p, vec_p)
    # rotated[1] = theta: the angel from z
    # rotate[2] = phi: the angel form x
    # print('rotated:',rotated)
    # rotated = np.subtract(vec_p,ref_p)
    # rotated[0] = vec_p[0]
    vec_rot = asCartesian(rotated)
    # print('vec_rot',vec_rot)

    return vec_rot, rotated


def vectors_angel(a, b):
    return cmath.asin(np.dot(a, b) / (cmath.sqrt(np.dot(a, a)) * cmath.sqrt(np.dot(b, b))))


def projection(a, b):
    # a: the main vector, the vector that other vectors will be added to
    # b: The second vector that will be added to the original vector
    ang = vectors_angel(a, b)
    vec_perp = b * cmath.cos(ang)
    vec_para = b * cmath.sin(ang)
    return vec_para, vec_perp


def axis_angels(vec):
    # print(vec)
    # l = vector_length(vec)
    # ang_x = np.arccos(vec[0]/l)
    ang_x = np.arctan(vec[1] / vec[2])
    if vec[2] == 0: ang_x = 0
    # ang_y = np.arccos(vec[1]/l)
    ang_y = np.arctan(vec[0] / vec[2])
    if vec[2] == 0: ang_y = 0
    # ang_z = np.arccos(vec[2]/l)
    ang_z = np.arctan(vec[0] / vec[1])
    if vec[1] == 0: ang_z = 0
    return ang_x, ang_y, ang_z


def rotate_refernce(vec):
    '''
    :param vec: the vector that
    :return:
    '''
    angx, angy, angz = axis_angels(vec)

    angx = angx
    angy = angy
    angz = angz

    print('ANGELS:\n', [np.degrees(x) for x in axis_angels(vec)])
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(angx), -np.sin(angx)],
        [0, np.sin(angx), np.cos(angx)]
    ])
    rot_y = np.array([
        [np.cos(angy), 0, np.sin(angy)],
        [0, 1, 0],
        [-np.sin(angy), 0, np.cos(angy)]
    ])
    rot_z = np.array([
        [np.cos(angz), -np.sin(angz), 0],
        [np.sin(angz), np.cos(angz), 0],
        [0, 0, 1]
    ])

    rot = np.dot(np.dot(rot_x, rot_y), rot_z)
    # rot = np.dot(rot_x,rot_y)
    return rot

#
#
# # vec_ref = np.array([4950.0, 9950.0, 20000.0])*1
# x = 20000
# # vec_ref = np.array([x,x*np.tan(np.deg2rad(45)),x*np.tan(np.deg2rad(30))])*1
# vec_ref = np.array([x,x,x*np.tan(np.deg2rad(30))])*-1
# vec = [0,0,15000]
#
# vec_rot, ang = np.array(rotate_vec(vec_ref,vec), dtype=float)
# # vec_rot = rotate_vec(vec,vec_ref)
#
# vec_rot2,ang2 = rotate_vec(vec_ref,vec_rot)
#
#
# #
# # # print('ROTATED\n',vec_rot)
# #
# # ----------------------------------------------------------
# # Visulization
# # Coordinates of the antenna and the source locations
#
# # Plotting in 3 dimensions
# fig = plt.figure(figsize=(10, 10))
# fig.suptitle('Please close the picture, when you finished', fontsize=20, color='brown')
# ax = plt.axes(projection='3d')
#
# plt.title('The overview of the antenna and source coordinates\n\n', fontsize=14)
# ax.scatter3D( vec_ref[0],vec_ref[1],vec_ref[2],
#              color='g'
#              )
# ax.plot(
#     [vec_ref[0], 0],
#     [vec_ref[1], 0],
#     [vec_ref[2], 0],
#     color='gold')
#
# ax.plot(
#     [vec[0], 0],
#     [vec[1], 0],
#     [vec[2]+10000, 0],
#     color='r')
#
# ax.plot(
#     [vec_rot[0], 0],
#     [vec_rot[1], 0],
#     [vec_rot[2], 0],
#     color='g')
#
# ax.plot(
#     [vec_rot2[0], 0],
#     [vec_rot2[1], 0],
#     [vec_rot2[2], 0],
#     color='b')
#
# ax.text(vec_ref[0] - 500, vec_ref[1], vec_ref[2] + 1000, 'The source\n', size=14, zorder=1,
#         color='r')
# ax.set_xlabel('X ')
# ax.set_ylabel('Y ')
# ax.set_zlabel('Z ')
# plt.show()  # to keep the picture until you close it
# plt.close(fig)  # Free the memory
#
