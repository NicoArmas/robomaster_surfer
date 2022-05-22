import _pickle
import json
import os.path
import pickle
import socket
from multiprocessing import Process

import matplotlib.pyplot as plt
import requests
import time

import cv2
import numpy as np

HEADERSIZE = 10


class FrameClient(Process):
    def __init__(self, host, port, frame_buffer, move_buffer, anomaly_buffer=None, logger=None):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.frame_buffer = frame_buffer
        self.move_buffer = move_buffer
        self.anomaly_buffer = anomaly_buffer
        self.logger = logger

    def run(self):
        self.logger.info('FrameClient started')
        self.connect()
        while True:
            while self.frame_buffer.empty():
                time.sleep(1 / 20)  # RM frame rate is 20fps
            frame = self.frame_buffer.get()

            if self.anomaly_buffer is not None:
                self.logger.info('getting anomaly map from server')
                res = self.get_anomaly_map(frame)
                if res is not None:
                    self.anomaly_buffer.put(res)

            self.logger.info('getting move from server')
            res = self.get_move(frame)
            if res is not None:
                self.move_buffer.put(res)

    def connect(self):
        if not self.connected:
            self.sock.connect((self.host, self.port))
            self.connected = True

    def disconnect(self):
        self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def send_packet(self, header, content):
        self.logger.info(f'sending packet {header}')
        packet = header + b'\t' + bytes(str(len(content)), 'utf-8') + b'\n' + content
        if not self.connected:
            self.connect()
        self.sock.sendall(packet)

    def get_response(self):
        if not self.connected:
            self.connect()

        res = self.sock.recv(HEADERSIZE)
        res_code = int(res.decode('utf-8'))

        if res_code == 200:
            data_length = int(self.sock.recv(HEADERSIZE).decode('utf-8'))
            self.logger.info(f'data length is {data_length}')
            data = b''
            while len(data) < data_length:
                data += self.sock.recv(1024)
            self.logger.info('received data')
            return pickle.loads(data)
        else:
            return None

    def get_move(self, frame):
        self.send_packet(b'get_movement', frame)
        self.logger.info('sent frame')
        res = self.get_response()
        self.disconnect()
        return res

    def get_anomaly_map(self, frame):
        self.send_packet(b'get_anomaly_map', frame)
        res = self.get_response()
        # res = cv2.imdecode(res, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(20, 20))
        plt.imshow(res, cmap='viridis')
        plt.savefig('anomaly_map.png')
        plt.close()
        print('anomaly map saved')
        self.disconnect()
        return res

    def __exit__(self):
        if self.connected:
            self.disconnect()

    # def frame_client(self):
    #     """
    #     It connects to the server, receives the message, unpickles it, and writes it to a file
    #
    #     :param host: The IP address of the server, defaults to 192.168.122.96 (optional)
    #     :param port: The port to which the server is listening, defaults to 5555 (optional)
    #     """
    #     count = 0
    #     full_msg = b''
    #     new_msg = True
    #     video_writer = cv2.VideoWriter('./frames/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
    #     try:
    #         while True:
    #             if new_msg:
    #                 sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #                 sock.connect((self.host, self.port))
    #             msg = sock.recv(1024)
    #             try:
    #                 if new_msg and msg:
    #                     msglen = int(msg[:HEADERSIZE])
    #                     new_msg = False
    #             except ValueError:
    #                 pass
    #
    #             full_msg += msg
    #
    #             if len(full_msg) - HEADERSIZE == msglen:
    #                 frame = pickle.loads(full_msg[HEADERSIZE:])
    #                 frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    #                 # cv2.imwrite(f'frames/frame{count}.jpg', frame)
    #                 video_writer.write(frame)
    #                 count += 1
    #                 new_msg = True
    #                 full_msg = b''
    #                 sock.close()
    #     except KeyboardInterrupt:
    #         video_writer.release()


if __name__ == '__main__':
    # if not os.path.exists('frames'):
    #     os.mkdir('frames')
    # frame_client()
    pass
