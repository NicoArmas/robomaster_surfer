import ctypes
import os
from multiprocessing import RawArray, Manager

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from .utils import FrameClient


# A camera is a device that can take pictures.
# It takes in a ROS message, converts it to a numpy array, and stores it in a buffer


class Camera:
    def __init__(self, node: Node, last_prediction,
                 save_data=False, stream_data=False, get_anomaly=False):
        """
        This function is called when the node is initialized. It creates a subscription to the camera topic and a
        callback function that is called when a new image is published

        :param node: the ROS node that we're using to subscribe to the camera topic
        :type node: Node
        :param framebuffer_size: The number of frames to store in the buffer
        :type framebuffer_size: int
        :param last_prediction: This is a shared array that is used to communicate the last prediction from the model
        :param save_data: If true, the node will save the data to a folder called data, defaults to False (optional)
        :param stream_data: If True, the node will stream the data to the server, defaults to False (optional)
        :param get_anomaly: If true, the anomaly buffer will be filled with the anomaly image, defaults to False (optional)
        """
        self.node = node
        self.save_data = save_data
        self.stream_data = stream_data
        shared_array = RawArray(ctypes.c_uint8, 49152)
        self.shared_array_np = np.ndarray(
            (128, 128, 3), dtype=np.uint8, buffer=shared_array)

        self.last_prediction = last_prediction
        self.frame_id = 0
        self.frame = None
        self.get_anomaly = get_anomaly

        if get_anomaly:
            self.anomaly_buffer = np.zeros((129, 128, 3), dtype=np.uint8)
        else:
            self.anomaly_buffer = None

        if self.stream_data:
            manager = Manager()
            self.new_frame = manager.Value('new_frame', False)
            self.frame_client = FrameClient('100.100.150.14', 5555, self.shared_array_np, self.last_prediction,
                                            self.new_frame, self.anomaly_buffer, self.node.get_logger())
            self.frame_client.start()

        if save_data and not os.path.exists('../../host/data'):
            os.mkdir('../../host/data')

        self.decoder = CvBridge()
        self.node.create_subscription(
            Image, 'camera/image_raw', self.camera_raw_callback, 1)

    async def camera_raw_callback(self, msg):
        """
        It takes the raw image from the camera, saves it to a buffer, and then saves it to a file if the user has
        requested it

        :param msg: the message that is received from the camera
        """
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.new_frame.value = True
        self.frame_id += 1

        if self.save_data:
            cv2.imwrite('./data/img_{}.png'.format(self.frame_id), self.frame)

        if self.stream_data:
            stream_frame = cv2.resize(self.frame, (128, 128))
            np.copyto(self.shared_array_np, stream_frame)

    def stop(self):
        """
        It closes the frame client and joins it
        """
        if self.stream_data:
            self.frame_client.close()
            self.frame_client.join()
