#!/usr/bin/env python

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform

import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image

import argparse
import time

# Import Openpose (Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath('../catkin_ws/src/openpose_pkg/src/openpose/build/python'))
from openpose import pyopenpose as op


class OpenPoseNode():
    def __init__(self, params, no_display):
        rospy.init_node('openpose_node')

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.no_display = no_display
        rospy.loginfo('OpenPose predictor is ready')

        # self.predictor = Predictor(weights_path)
        self.bridge = CvBridge()
        self.img_subscriber = rospy.Subscriber('pose_detection_images', Image, self.predict_pose, queue_size=10)
        self.det_publisher = rospy.Publisher('detection', String, queue_size=10)
        rospy.loginfo('OpenPose node is done')

    def predict_pose(self, image_msg: Image):
        rospy.loginfo('Received image')
        current_time = time.time()
        image = self.bridge.imgmsg_to_cv2(image_msg, 'rgb8')
        # image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
        # current_time = time.time()
        
        datum = op.Datum()
        # imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if not self.no_display:
            cv2.imshow("OpenPose 1.7.0", datum.cvOutputData)
            key = cv2.waitKey(15)
            # if key == 27: break

        n_people = datum.poseKeypoints.shape[0]
        passed_time = time.time() - current_time
        rospy.loginfo(f'Detected {n_people} people, this took {passed_time} seconds')
        self.det_publisher.publish(f'Detected {n_people} people')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", default="../catkin_ws/src/openpose_pkg/src/openpose/examples/media/", 
                                help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("--no_display", default=False, 
                                help="Enable to disable the visual display.")
        args = parser.parse_known_args()

        params = dict()
        params["model_folder"] = "../catkin_ws/src/openpose_pkg/src/openpose/models/"
        # params["net_resolution"] = "-512x256"
        params["net_resolution"] = "-256x128"

        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        openpose_node = OpenPoseNode(params, args[0].no_display)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
