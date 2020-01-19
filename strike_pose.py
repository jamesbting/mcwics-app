from __future__ import print_function, division
import argparse
import cv2
import wrnchAI
from visualizer import Visualizer
from utils import videocapture_context
import numpy as np
import serial
import vectorProcessing as vp
import time
from PIL import Image

webcam = 0
licenseKey = "7DBE74-87C9F6-4F179D-B703AB-05C75F-5A721D"
imgInd = 1 #initial index of image


def main():
    ser1 = serial.Serial('COM9', 9600)
    code = wrnchAI.license_check_string(licenseKey)
    time.sleep(4)
    ser1.write('0'.encode())

    # setting tracking parameters
    params = wrnchAI.PoseParams()
    params.bone_sensitivity = wrnchAI.Sensitivity.high
    params.joint_sensitivity = wrnchAI.Sensitivity.high
    params.enable_tracking = True
    # create resolution
    params.preferred_net_width = 500
    params.preferred_net_height = 500
    # format output
    output = wrnchAI.JointDefinitionRegistry.get('j23')
    print('Initializing networks')
    estimate = wrnchAI.PoseEstimator(models_path='..\\..\\..\\..\\bin\\wrModels', license_string=licenseKey,
                                     params=params, gpu_id=0, output_format=output)
    print('Initialization done!')
    options = wrnchAI.PoseEstimatorOptions()
    print('Turning on webcam now')


    # run-time vars:
    pose_img_path = r"C:\Users\tayse\OneDrive\Desktop\wrnchAI-engine-GPU-1.15.0-Windows-amd64\src\wrnchAI\wrSamples\python\Assets\pic.jpg"
    # extract McWICS Letters:
    #pose_imgs = np.load(pose_img_path)
    #pose_imgs_list = [pose_imgs[i].tolist() for i in range(6)]
    pose = vp.request_joint_data(pose_img_path)

    with videocapture_context(0) as cap:
        visual = Visualizer()
        joint_def = estimate.human_2d_output_format()
        bone_pairs = joint_def.bone_pairs()
        stagger = 0
        while True:
            #counter_letter = 0
            _, frame = cap.read()
            if frame is not None:
                estimate.process_frame(frame, options)
                humans2d = estimate.humans_2d()
                visual.draw_image(frame)
                for human in humans2d:
                    visual.draw_points(human.joints())
                    visual.draw_lines(human.joints(), bone_pairs)
                    if stagger % 15 == 0:
                        # gives the coordinates to calculate similarity
                        input_data = human.joints().tolist()  # list of floats
                        comp_val = vp.compare_dance(input_data, pose)  # compare results
                        if 0 <= comp_val <= 0.08:
                            # green
                            ser1.write('3'.encode())
                            print('excellent')
                            print(comp_val)
                            print("GET READY FOR NEXT LETTER!")
                        elif 0.08 < comp_val <= 0.14:
                            # green
                            ser1.write('2'.encode())
                            print('good')
                            print(comp_val)
                        elif 0.14 < comp_val:
                            # red
                            ser1.write('1'.encode())
                            print('bad')
                            print(comp_val)
                    if stagger is 15:
                        stagger = 0
                    stagger += 1

                visual.show()

            key = cv2.waitKey(1)
            if key & 255 == 27:
                break


if __name__ == '__main__':
    main()
