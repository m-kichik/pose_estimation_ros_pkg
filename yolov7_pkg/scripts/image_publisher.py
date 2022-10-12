#!/<path to your interpreter [env]> python

import sys
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def img_publisher(img_path):
    rospy.init_node('img_publisher', anonymous=True)
    pub = rospy.Publisher('pose_detection_images', Image, queue_size=10)

    bridge = CvBridge()

    image = cv2.imread(img_path)
    image_msg = bridge.cv2_to_imgmsg(image, 'rgb8')

    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        pub.publish(image_msg)
        rospy.loginfo(f'published {img_path.split("/")[-1]}')
        rate.sleep()



if __name__ == '__main__':
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
    else:
        sys.exit(1)
    try:
        img_publisher(img_path)
    except rospy.ROSInterruptException:
        pass
