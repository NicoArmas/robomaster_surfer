import pickle
import socket
import time
import cv2
from multiprocessing import Process

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
        self.idx = 0

    def run(self):
        """
        It waits for a frame to be available in the frame buffer, then sends it to the server and waits for a response
        """
        self.connect()
        while True:
            while self.frame_buffer.empty():
                time.sleep(1 / 20)  # RM frame rate is 20fps
            frame = self.frame_buffer.get()

            # if self.anomaly_buffer is not None:
            #     res = self.get_anomaly_map(frame)
            #     if res is not None:
            #         self.anomaly_buffer.put(res)

            res = self.get_move(frame)
            res = (int(res[0]), int(res[1]), int(res[2]))
            if res is not None:
                self.move_buffer.put(res)

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
        packet = header + b'\t' + bytes(str(len(content)), 'utf-8') + b'\n' + content
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

        if res_code == 200:
            data_length = int(self.sock.recv(HEADERSIZE).decode('utf-8'))

            data = b''
            while len(data) < data_length:
                data += self.sock.recv(1024)

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

        res = self.get_response()
        self.disconnect()
        return res

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

        self.disconnect()
        return res

    def close(self) -> None:
        """
        The function closes the connection to the server and logs the action
        """
        self.disconnect()

        super().close()

    def __exit__(self):
        """
        If the object is connected, disconnect it
        """
        if self.connected:
            self.disconnect()
