import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge


# A camera is a device that can take pictures.
class Camera:
    def __init__(self, node: Node, framebuffer_size: int):
        """
        This function initializes the class with the node and framebuffer size

        :param node: The node that will be used to subscribe to the camera topic
        :type node: Node
        :param framebuffer_size: The number of frames to store in the framebuffer
        :type framebuffer_size: int
        """
        self.node = node
        self.framebuffer = np.zeros((framebuffer_size, 720, 1280, 3), dtype=np.uint8)
        self.frame_idx = 0
        self.frame = None
        self.decoder = CvBridge()

    def camera_callback(self, msg):
        """
        It takes in a ROS message, converts it to a numpy array, and stores it in a buffer

        :param msg: the image message
        """
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.framebuffer[self.frame_idx] = self.frame
        self.frame_idx = (self.frame_idx + 1) % self.framebuffer.shape[0]
