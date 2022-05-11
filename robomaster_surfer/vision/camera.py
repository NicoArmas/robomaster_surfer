from copy import deepcopy

import cv2
import numpy as np
import robomaster_msgs.msg
import libmedia_codec
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os


# A camera is a device that can take pictures.
# It takes in a ROS message, converts it to a numpy array, and stores it in a buffer
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
        self.framebuf_idx = 0
        self.frame_id = 0
        self.frame = None
        self.buffer_full = False
        self.video_idx = 0
        self._video_decoder = libmedia_codec.H264Decoder()
        if not os.path.exists('./data'):
            os.mkdir('./data')
        self.decoder = CvBridge()
        # Try this topic: camera/image_h264
        self.node.create_subscription(Image, 'camera/image_raw', self.camera_raw_callback, 1)
        # self.node.create_subscription(robomaster_msgs.msg.H264Packet, 'camera/image_h264', self.camera_callback, 1)

    async def camera_raw_callback(self, msg):
        """
        It takes in a ROS message, converts it to a numpy array, and stores it in a buffer

        :param msg: the image message
        """
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.framebuffer[self.framebuf_idx] = self.frame
        self.framebuf_idx = (self.framebuf_idx + 1) % self.framebuffer.shape[0]
        # self.node.get_logger().debug("Frame {} received".format(self.frame_id))
        # cv2.imwrite(self.path.format(self.frame_id), cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        # self.node.get_logger().debug("Frame {} saved".format(self.frame_id))
        self.frame_id += 1
        if self.framebuf_idx == 0:
            self.buffer_full = True

    def save_buffer(self, filename: str):
        """
        It takes the frames from the buffer and saves them to a video file

        :param filename: the name of the file to save the video to
        :type filename: str
        """
        self.buffer_full = False
        w = cv2.VideoWriter(filename.format(self.video_idx), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
        for i, frame in enumerate(deepcopy(self.framebuffer)):
            w.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(path.format(i), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.node.get_logger().info(f'Saved buffer frames to {filename.format(self.video_idx)}')
        w.release()
        self.video_idx += 1

    def _h264_decode(self, data):
        res_frame_list = []
        frames = self._video_decoder.decode(data)
        for frame_data in frames:
            (frame, width, height, ls) = frame_data
            if frame:
                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((height, width, 3)))
                res_frame_list.append(frame)
        return res_frame_list

    async def camera_callback(self, msg):
        self.frame = msg
        self.node.get_logger().info(f'got frame: {msg}')
        in_frame = (
        np
        .frombuffer(msg.data, np.uint8)
        .reshape([height, width, 3])
    )
        exit(0)
