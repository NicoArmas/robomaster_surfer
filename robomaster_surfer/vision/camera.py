import socketserver
from copy import deepcopy

import cv2
import numpy as np
import robomaster_msgs.msg
import libmedia_codec
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from .utils import FrameClient
import os
import multiprocessing
from multiprocessing import Queue
import threading

# A camera is a device that can take pictures.
# It takes in a ROS message, converts it to a numpy array, and stores it in a buffer


class Camera:
    def __init__(self, node: Node, framebuffer_size: int, save_data=False, save_video=False, save_preds=False):
        """
        This function initializes the class with the node and framebuffer size

        :param node: The node that will be used to subscribe to the camera topic
        :type node: Node
        :param framebuffer_size: The number of frames to store in the framebuffer
        :type framebuffer_size: int
        """
        self.node = node
        self.save_data = save_data
        self.buf_size = framebuffer_size
        self.framebuffer = np.zeros((framebuffer_size, 720, 1280, 3), dtype=np.uint8)
        self.streambuffer = Queue()
        self.anomaly_buffer = Queue()
        self.move_buffer = Queue()
        self.framebuf_idx = 0
        self.frame_id = 0
        self.frame = None
        self.buffer_full = [False] * 2
        self.video_idx = 0
        self.save_video = save_video
        self.video_writer = None
        self.save_preds = save_preds
        self.prediction = None
        self._video_decoder = libmedia_codec.H264Decoder()
        self.frame_client = FrameClient('100.99.238.59', 5555, self.streambuffer,
                                        self.anomaly_buffer, self.move_buffer, logger=self.node.get_logger())
        self.frame_client.start()

        if not os.path.exists('./data'):
            os.mkdir('./data')

        if self.save_video:
            self.node.get_logger().info("Initializing video writer")
            self.video_writer = cv2.VideoWriter('./data/video.mp4',
                                                cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))

        self.decoder = CvBridge()
        # Try this topic: camera/image_h264
        self.node.create_subscription(Image, 'camera/image_raw', self.camera_raw_callback, 1)
        # self.node.create_subscription(robomaster_msgs.msg.H264Packet, 'camera/image_h264', self.camera_callback, 1)

    async def camera_raw_callback(self, msg):
        """
        It takes in a ROS message, converts it to a numpy array, and stores it in a buffer

        :param msg: the image message
        """
        self.frame = self.decoder.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.framebuffer[self.framebuf_idx] = self.frame
        self.framebuf_idx = (self.framebuf_idx + 1) % self.framebuffer.shape[0]
        if self.save_data:
            cv2.imwrite('./data/img_{}.png'.format(self.frame_id), self.frame)
        if self.save_video:
            self.video_writer.write(self.frame)
            stream_frame = cv2.resize(self.frame, (128, 128))
            # stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2GRAY)
            stream_frame = cv2.imencode('.png', stream_frame)[1].dumps()
            self.streambuffer.put(stream_frame)
        # self.node.get_logger().debug("Frame {} received".format(self.frame_id))
        # cv2.imwrite(self.path.format(self.frame_id), cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        # self.node.get_logger().debug("Frame {} saved".format(self.frame_id))
        self.frame_id += 1
        # if self.framebuf_idx == 0:
        #     self.buffer_full = True
        #     self.node.get_logger().info(f"Saving buffer {self.framebuffer_used}")
        #     run save buffer in another process
        # multiprocessing.Process(target=Camera.save_buffer, args=(self, 'data/video{}.mp4')).start()

    def save_buffer(self, filename: str):
        """
        It takes the frames from the buffer and saves them to a video file

        :param filename: the name of the file to save the video to
        :type filename: str
        """
        buffer_to_empty = self.framebuffers[np.where(self.buffer_full == True)[0][0]]
        w = cv2.VideoWriter(filename.format(self.video_idx), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
        for frame in self.framebuffer[buffer_to_empty]:
            w.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(path.format(i), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.node.get_logger().info(f'Saved buffer frames to {filename.format(self.video_idx)}')
        w.release()
        self.buffer_full[buffer_to_empty] = False
        self.node.is_saving = False
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
                .reshape([1280, 720, 3])
        )
        exit(0)

    def stop(self):
        if self.save_video:
            self.video_writer.release()
            self.frame_client.close()
            self.frame_client.join()
