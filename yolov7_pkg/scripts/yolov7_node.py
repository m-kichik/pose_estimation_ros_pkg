#!/<path to your interpreter [env]> python

import os
import sys

import rospy
import cv2
import torch
from torchvision import transforms
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image


yolov7_module_path = os.path.abspath(os.path.join('/<path to your package>/yolov7_pkg/scripts/yolov7'))
if yolov7_module_path not in sys.path:
    sys.path.append(yolov7_module_path)

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def visualize(output, image):
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg


def output2dict(output):
    dets = []
    for det in output:
        dets.append({'nose': det[7:10],
                     'right eye': det[10:13],
                     'left eye': det[13:16],
                     'right ear': det[16:19],
                     'left ear': det[19:22],
                     'right shoulder': det[22:25],
                     'left shoulder': det[25:28],
                     'right elbow': det[28:31],
                     'left elbow': det[31:34],
                     'right hand': det[34:37],
                     'left hand': det[37:40],
                     'right hip': det[40:43],
                     'left hip': det[43:46],
                     'right knee': det[46:49],
                     'left knee': det[49:52],
                     'right foot': det[52:55],
                     'left foot': det[55:58],
                     })
    return dets


def plot_hand_kpts(im, output_dict):
    kpt_color = (0, 0, 240)
    radius = 7

    right_x, right_y, right_conf = output_dict['right hand']
    left_x, left_y, left_conf = output_dict['left hand']

    if right_conf >= 0.5:
        cv2.circle(im, (int(right_x), int(right_y)), radius, kpt_color, -1)

    if left_conf >= 0.5:
        cv2.circle(im, (int(left_x), int(left_y)), radius, kpt_color, -1)


def draw_hand_detections(image, output_dict):
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    for idx in range(len(output_dict)):
        plot_hand_kpts(nimg, output_dict[idx])

    return nimg


class Predictor():
    """
    Predict pose with YOLOv7 (cv2 image required)
    """

    def __init__(self, weights_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weigths = torch.load(weights_path, map_location=self.device)
        self.model = self.weigths['model']
        _ = self.model.float().eval()

        if torch.cuda.is_available():
            self.model.half().to(self.device)

        rospy.loginfo('YOLOv7 predictor is ready')

    def inference(self, image):
        # image = cv2.imread(image_path)
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(self.device)
        with torch.no_grad():
            output, _ = self.model(image)
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'],
                                             kpt_label=True)
            output = output_to_keypoint(output)
        return output, image


class YOLOv7Node():
    def __init__(self):
        rospy.init_node('YOLOv7_pose_node')

        weights_path = '../catkin_ws/src/yolov7_pkg/src/weights/yolov7-w6-pose.pt'
        self.predictor = Predictor(weights_path)

        self.bridge = CvBridge()
        self.img_subscriber = rospy.Subscriber('pose_detection_images', Image, self.predict_pose, queue_size=10)
        self.det_publisher = rospy.Publisher('detection', String, queue_size=10)

        rospy.loginfo('YOLOv7 node is done')

    def predict_pose(self, image_msg: Image):
        rospy.loginfo('Received image')

        image = self.bridge.imgmsg_to_cv2(image_msg, 'rgb8')
        output, _ = self.predictor.inference(image)
        n_people = output.shape[0]

        rospy.loginfo(f'Detected {n_people} people')
        self.det_publisher.publish(f'Detected {n_people} people')


if __name__ == '__main__':
    try:
        yolov7_node = YOLOv7Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
