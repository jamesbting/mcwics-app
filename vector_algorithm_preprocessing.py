# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:39:23 2020

@author: yasha
"""
import time
import requests
import json
import os
import numpy as np
import math

# helper methods:
def json_r(filename):
    with open(filename) as f:
        return(json.load(f))
def unit(np_array):
    return np_array / np.linalg.norm(np_array)
"""
def cos_sim(v, u):
    return (np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(u)))
"""
def split_joints_tuple(joints_array):
    return list(zip(joints_array[::2], joints_array[1::2]))
def squish(x):
    return (1 - math.e**(-x))

## 0.8 threshhold value?

# comparison algorithm
def compare_dance(init_pose, sample_pos):
    zip_init_pose = split_joints_tuple(init_pose)
    zip_sample_pos = split_joints_tuple(sample_pos)
    accum = 0; num_vecs = 0
    for j in range(len(init_pose) // 2):
        for k in range(len(init_pose) // 2):
            if j >= k:
                continue
            elif zip_init_pose[j][0] < 0 or zip_sample_pos[j][0] < 0:
                continue
            else: 
                v_init_j = unit(np.array(zip_init_pose[j])); v_init_k = unit(np.array(zip_init_pose[k]))
                v_sam_j = unit(np.array(zip_sample_pos[j])); v_sam_k = unit(np.array(zip_sample_pos[k]))
                v_init_diff = (v_init_j - v_init_k)
                v_sam_diff = (v_sam_j - v_sam_k)
                accum += np.linalg.norm(v_init_diff - v_sam_diff)
                num_vecs+=1
    return(squish(accum / num_vecs))

"""
Returns a parsed list of joint data from cloud API
"""
def request_joint_data(filepath):
    LOGIN_URL = 'https://api.wrnch.ai/v1/login'
    JOBS_URL = 'https://api.wrnch.ai/v1/jobs'
    API_KEY = "f2b6d23f-b864-4c59-bbb3-f5aae914fb3b"
    
    resp_auth = requests.post(LOGIN_URL, data={'api_key':API_KEY})
    JWTTOKEN = json.loads(resp_auth.text)['access_token']
    
    # load
    with open(filepath, 'rb') as f:
        resp_sub_job = requests.post(JOBS_URL,
                                 headers={'Authorization':f'Bearer {JWTTOKEN}'},
                                 files={'media':f},
                                 data={'work_type':'json'})
    job_id = json.loads(resp_sub_job.text)['job_id']
    print('Status code:',resp_sub_job.status_code)
    print('Response:',resp_sub_job.text)
    
    time.sleep(0.5)
    GET_JOB_STATUS_URL = 'https://api.wrnch.ai/v1/status' + '/' + job_id
    resp_get_job = requests.get(GET_JOB_STATUS_URL,headers={'Authorization':f'Bearer {JWTTOKEN}'})
    print('Status code:',resp_get_job.status_code)
    print('\nResponse:',resp_get_job.text)    

    time.sleep(5)
    GET_JOB_URL = JOBS_URL + '/' + job_id
    print(GET_JOB_URL)
    resp_get_job = requests.get(GET_JOB_URL,headers={'Authorization':f'Bearer {JWTTOKEN}'})
    cloud_pose_estimation = json.loads(resp_get_job.text)
    return cloud_pose_estimation['frames'][0]['persons'][0]['pose2d']['joints']

def make_base_model(pathname, posename):
    imgs = os.listdir(pathname)
    i = 0
    while(i < len(imgs)):
        data = request_joint_data(imgs[i])
        if i == 0:
            output = np.array(data)
        else:
            vec = np.array(data)
            output = np.vstack((output, vec))
        i+=1
    # writing to binary npy final
    np.save(posename, output) # posename = <posename>.npy

if __name__ == "__main__":
    # importing .json/jpeg file
    os.chdir(r"C:\Users\yasha\Desktop\McWics\james-test\base")
    imgs = os.listdir(r"C:\Users\yasha\Desktop\McWics\james-test\base")
    i = 0
    while(i <= 9):
        data1 = request_joint_data(imgs[i])
        data2 = request_joint_data(imgs[i+1])
        print("result of comparison: {}".format(compare_dance(data1,data2)))
        i+=2
    data = request_joint_data(imgs[0]); data_ = request_joint_data(imgs[6])
    print(compare_dance(data, data_))
    #print(compare_dance((0.5,0.43,0.42,0.64,0.23,0.88,0.76, 0.43, 0.6, 0.32),(0.5335,0.4253,0.235,0.273,0.769,0.67,0.47, 0.457, 0.543, 0.56)))
    #print(compare_dance(joints, joints))
    
   
