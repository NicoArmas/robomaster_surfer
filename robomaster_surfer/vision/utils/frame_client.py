import pickle
import socket
import time
import cv2
from multiprocessing import Process

from numpy import argmax

HEADERSIZE = 10


class FrameClient(Process):
    def __init__(self, host, port, frame_buffer, move_buffer, anomaly_buffer=None, logger=None):
        """
        This function initializes the class by setting the host, port, socket, connected, frame_buffer, move_buffer,
        anomaly_buffer, logger, and idx variables

        :param host: the IP address of the server
        :param port: the port number to connect to
        :param frame_buffer: a queue of frames to be sent to the client
        :param move_buffer: a queue of moves to send to the server
        :param anomaly_buffer: This is a buffer that will be used to store the anomaly data
        :param logger: a logger object to log messages to
        """
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.frame_buffer = frame_buffer
        self.move_buffer = move_buffer
        self.anomaly_buffer = anomaly_buffer
        self.logger = logger
        self.idx = 0

    def run(self):
        """
        It waits for a frame to be available in the frame buffer, then sends it to the server and waits for a response
        """
        self.logger.info('FrameClient started')
        self.connect()
        while True:
            frame = self.frame_buffer[:]
            frame = cv2.imencode('.png', frame)[1].dumps()

            # if self.anomaly_buffer is not None:
            #     self.logger.info('getting anomaly map from server')
            #     res = self.get_anomaly_map(frame)
            #     if res is not None:
            #         self.anomaly_buffer.put(res)

            #self.logger.info('getting move from server')
            res = self.get_move(frame)
            self.logger.info("Received: "+str(res))
            if res is not None:
                self.move_buffer.value = int(res)

    def connect(self):
        """
        If the socket is not connected, connect it to the host and port specified in the constructor
        """
        if not self.connected:
            self.sock.connect((self.host, self.port))
            self.connected = True

    def disconnect(self):
        """
        It closes the socket and then creates a new socket.
        """
        self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def send_packet(self, header, content):
        """
        It sends a packet to the server

        :param header: the header of the packet, which is a string
        :param content: the content of the packet
        """
        #self.logger.info(f'sending packet {header}')
        packet = header + b'\t' + \
            bytes(str(len(content)), 'utf-8') + b'\n' + content
        if not self.connected:
            self.connect()
        self.sock.sendall(packet)

    def get_response(self):
        """
        It receives a response from the server, and if the response code is 200, it receives the data and unpickles it
        :return: The data is being returned as a pickled object.
        """
        if not self.connected:
            self.connect()

        res = self.sock.recv(HEADERSIZE)
        res_code = int(res.decode('utf-8'))
        # self.logger.info(str(res_code))

        if res_code == 200:
            data_length = int(self.sock.recv(HEADERSIZE).decode('utf-8'))
            #self.logger.info(f'data length is {data_length}')
            data = b''
            while len(data) < data_length:
                data += self.sock.recv(1024)
            #self.logger.info('received data')
            # self.logger.info(str(data))
            return data
        else:
            return None

    def get_move(self, frame):
        """
        It sends a frame to the server, waits for a response, and then disconnects

        :param frame: a numpy array of shape (1, 3, 84, 84)
        :return: The response from the server.
        """
        self.send_packet(b'get_movement', frame)
        #self.logger.info('sent frame')
        res = self.get_response()
        self.disconnect()
        return res.decode()

    def get_anomaly_map(self, frame):
        """


        :param frame: the frame to be processed
        :return: The anomaly map is being returned.
        """
        self.send_packet(b'get_anomaly_map', frame)
        res = self.get_response()
        res = cv2.imdecode(res, cv2.IMREAD_COLOR)
        cv2.imwrite(f'./data/anomaly_map_{self.idx}.png', res)
        self.idx += 1
        #self.logger.info('Saved anomaly map')
        self.disconnect()
        return res

    def close(self) -> None:
        """
        The function closes the connection to the server and logs the action
        """
        self.disconnect()
        #self.logger.info('FrameClient closed')
        super().close()

    def __exit__(self):
        """
        If the object is connected, disconnect it
        """
        if self.connected:
            self.disconnect()
