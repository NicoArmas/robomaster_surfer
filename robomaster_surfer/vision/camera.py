from copy import deepcopy

import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


# A camera is a device that can take pictures.
class Camera:
    def __init__(self, node: Node, framebuffer_size: int, save_data=False,
                 path="frame_{}.png"):
        """
        This function initializes the class with the node and framebuffer size

        :param node: The node that will be used to subscribe to the camera topic
        :type node: Node
        :param framebuffer_size: The number of frames to store in the framebuffer
        :type framebuffer_size: int
        """
        self.node = node
        # self.framebuffer = np.zeros((framebuffer_size, 720, 1280, 3), dtype=np.uint8)
        # self.framebuf_idx = 0
        self.frame_id = 0
        self.frame = None
        self.buffer_full = False
        self.save_data = save_data
        self.decoder = CvBridge()
        self.path = path
        self.node.create_subscription(Image, 'camera/image_raw', self.camera_callback, 1)

    async def camera_callback(self, msg):
        """
        It takes in a ROS message, converts it to a numpy array, and stores it in a buffer

        :param msg: the image message
        """
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # self.framebuffer[self.framebuf_idx] = self.frame
        # self.framebuf_idx = (self.framebuf_idx + 1) % self.framebuffer.shape[0]
        self.node.get_logger().debug("Frame {} received".format(self.frame_id))
        if self.save_data:
            cv2.imwrite(self.path.format(self.frame_id), cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
            self.node.get_logger().debug("Frame {} saved".format(self.frame_id))
        self.frame_id += 1
        # if self.framebuf_idx == 0:
        #     self.buffer_full = True

    # def save_buffer(self, path: str):
    #     self.buffer_full = False
    #     # w = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
    #     for i, frame in enumerate(deepcopy(self.framebuffer)):
    #         # w.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #         cv2.imwrite(path.format(i), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     self.node.get_logger().info(f'Saved buffer frames to {path}')
    #     # w.release()
