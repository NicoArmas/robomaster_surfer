import socket
import time
from multiprocessing import Process

import cv2

HEADERSIZE = 10


class FrameClient(Process):
    def __init__(self, host, port, frame_buffer, move_buffer, new_frame, anomaly_buffer=None, logger=None):
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
        self.new_frame = new_frame

    def run(self):
        self.logger.info('FrameClient started')
        self.connect()
        while True:
            if self.new_frame.value:
                self.new_frame.value = False
                frame = self.frame_buffer[:]
                frame = cv2.imencode('.png', frame)[1].dumps()

                # self.logger.info('getting move from server')
                res = self.get_move(frame)
                self.logger.info("Received: " + str(res))
                if res is not None:
                    self.move_buffer.value = int(res)

                if self.anomaly_buffer is not None:
                    res = self.get_anomaly_map(frame).decode()
                    if res is not None:
                        self.anomaly_buffer.value = res
            else:
                time.sleep(0.1)

    def connect(self):
        if not self.connected:
            self.sock.connect((self.host, self.port))
            self.connected = True

    def disconnect(self):
        self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def send_packet(self, header, content):
        # self.logger.info(f'sending packet {header}')
        packet = header + b'\t' + \
                 bytes(str(len(content)), 'utf-8') + b'\n' + content
        if not self.connected:
            self.connect()
        self.sock.sendall(packet)

    def get_response(self):
        if not self.connected:
            self.connect()

        res = self.sock.recv(HEADERSIZE)
        res_code = int(res.decode('utf-8'))
        # self.logger.info(str(res_code))

        if res_code == 200:
            data_length = int(self.sock.recv(HEADERSIZE).decode('utf-8'))
            # self.logger.info(f'data length is {data_length}')
            data = b''
            while len(data) < data_length:
                data += self.sock.recv(1024)
            # self.logger.info('received data')
            # self.logger.info(str(data))
            return data
        else:
            return None

    def get_move(self, frame):
        self.send_packet(b'get_movement', frame)
        # self.logger.info('sent frame')
        res = self.get_response()
        self.disconnect()
        return res.decode()

    def get_anomaly_map(self, frame):
        self.send_packet(b'get_anomaly_map', frame)
        res = self.get_response()
        res = cv2.imdecode(res, cv2.IMREAD_COLOR)
        cv2.imwrite(f'./data/anomaly_map_{self.idx}.png', res)
        self.idx += 1
        # self.logger.info('Saved anomaly map')
        self.disconnect()
        return res

    def close(self) -> None:
        self.disconnect()
        # self.logger.info('FrameClient closed')
        super().close()

    def __exit__(self):
        if self.connected:
            self.disconnect()
