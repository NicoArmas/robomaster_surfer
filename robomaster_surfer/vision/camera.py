import rclpy
from geometry_msgs.msg import Twist, Pose
from rclpy.node import Node
from cv_bridge import CvBridge


class Camera:
    def __init__(self, node: Node):
        self.node = node
        self.frame = None
        self.decoder = CvBridge()

    def camera_callback(self, msg):
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="rgb8")
