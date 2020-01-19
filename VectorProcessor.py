# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:39:23 2020

@author: yasha
"""
import requests
import json
import os
import numpy as np
import math


class VectorProcessor:

    def __init__(self,filename):
        # attributes
        self.base_set =
        self.sample_set =

    # helper methods:

    # loads the json file and returns a
    def json_r(self, filename):
        with open(filename) as f:
            return json.load(f)

    # function that returns a unit vector
    def unit(self,np_array):
        return np_array / np.linalg.norm(np_array)

    def split_joints_tuple(self,joints_array):
        return list(zip(joints_array[::2], joints_array[1::2]))

    # function that squishes values to numbers between 0 and 1, and is essentially
    # a mesuremnt of how similar the moves are
    def squish(self,x):
        return 1 - math.e ** (-x)

    ## 0.8 threshhold value?

    # comparison algorithm
    def compare_dance(self, init_pose, sample_pos):
        zip_init_pose = self.split_joints_tuple(init_pose)
        zip_sample_pos = self.split_joints_tuple(sample_pos)
        accum = 0
        for j in range(len(init_pose) // 2):
            for k in range(len(init_pose) // 2):
                if j >= k:
                    continue
                else:
                    v_init_j = self.unit(np.array(zip_init_pose[j]));
                    v_init_k = self.unit(np.array(zip_init_pose[k]))
                    v_sam_j = self.unit(np.array(zip_sample_pos[j]));
                    v_sam_k = self.unit(np.array(zip_sample_pos[k]))
                    v_init_diff = (v_init_j - v_init_k)
                    v_sam_diff = (v_sam_j - v_sam_k)
                    accum += self.squish(np.linalg.norm(v_init_diff - v_sam_diff))
        return (accum)


    # testing the class, ignorexz
    if __name__ == "__main__":
        # importing .json file
        os.chdir(r"C:\Users\yasha\Desktop\McWics")
        pose_est = json_r('json-IMG_20190729_160008620.json')

        # looking at data
        # print(str(pose_est['frames'])[:300])
        pose = pose_est['frames'][0]['persons'][0]
        size_pose = len(pose['pose2d']['joints'])
        print(size_pose)
        joints = pose['pose2d']['joints']
        # joint 0 x,y (ranges from 0 - 49)
        (x1, y1) = (pose['pose2d']['joints'][0], pose['pose2d']['joints'][1])
        (xN, yN) = (pose['pose2d']['joints'][48], pose['pose2d']['joints'][49])

        r = (1, 2, 3, 4)
        print(compare_dance((0.5, 0.43, 0.42, 0.64, 0.23, 0.88, 0.76, 0.43, 0.6, 0.32),
                            (0.5335, 0.4253, 0.235, 0.273, 0.769, 0.67, 0.47, 0.457, 0.543, 0.56)))
        print(compare_dance(joints, joints))
