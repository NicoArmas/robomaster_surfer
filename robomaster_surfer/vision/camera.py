from copy import deepcopy

import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


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
        self.buffer_full = False
        self.decoder = CvBridge()
        self.node.create_subscription(Image, 'camera/image_raw', self.camera_callback, 1)

    async def camera_callback(self, msg):
        """
        It takes in a ROS message, converts it to a numpy array, and stores it in a buffer

        :param msg: the image message
        """
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.framebuffer[self.frame_idx] = self.frame
        self.frame_idx = (self.frame_idx + 1) % self.framebuffer.shape[0]
        if self.frame_idx == 0:
            self.buffer_full = True

    def save_video(self, filename: str):
        self.buffer_full = False
        w = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'avc1'), 20, (1280, 720))
        for frame in deepcopy(self.framebuffer):
            w.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.node.get_logger().info('Saved video to output.mp4')
        w.release()
