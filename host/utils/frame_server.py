import logging
import pickle
import socketserver
from typing import Callable, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from host.obstacle_avoidance.obstacle_avoidance import ObstacleAvoidance

HEADERSIZE = 10


class FrameServer(socketserver.BaseServer):
    def __init__(self,
                 server_address: Tuple[str, int],
                 RequestHandlerClass: Callable[..., socketserver.StreamRequestHandler]):
        super().__init__(server_address, RequestHandlerClass)


def pack_response(code: int, data: bytes):
    """
    It takes a return code and a data payload, and returns a response
    packet with the return code and data length in the header

    :param code: the return code of the request
    :type code: int
    :param data: the data to be sent
    :type data: bytes
    :return: The return code and the length of the data.
    """
    res = bytes(f'{code:<{HEADERSIZE}}', 'utf-8') + \
          bytes(f'{len(data):<{HEADERSIZE}}', 'utf-8') + \
          data
    # logging.info(f'Sending response, return code: {res[:10].split()[0]}, length: {res[11:20].split()[0]}')
    return res


class FrameHandler(socketserver.StreamRequestHandler):

    def send_error(self, code: int = 404):
        """
        It sends an error response to the client

        :param code: The HTTP status code to send, defaults to 404
        :type code: int (optional)
        """
        self.wfile.write(pack_response(code, b'Error'))

    def handle(self):
        """
        It receives a request from the client, processes it, and sends a response back
        :return: The response is the predicted movement of the object in the frame.
        """
        logging.info(f'Got connection from {self.client_address}')
        msg = self.rfile.readline().split()
        logging.info(msg)
        req = msg[0].decode('utf-8')
        content_size = int(msg[1].decode('utf-8'))
        if not req:
            return
        content = self.rfile.read(content_size)
        frame = cv2.imdecode(pickle.loads(content), cv2.IMREAD_COLOR)
        cv2.imwrite('frame.png', frame)
        frame = self.server.transform(frame).to('cuda')

        if req == 'get_movement':
            response = self.server.model(torch.unsqueeze(frame, 0))
            response = torch.softmax(response.cpu().detach(), axis=1)
            response = response.argmax(axis=1)
            response = response.item()
            logging.info(str(response))
            self.wfile.write(pack_response(200, str(response).encode('utf-8')))

        elif req == 'get_anomaly_map':
            self.wfile.write(pack_response(404, b'Not Found'))
            self.server.model.eval()
            _, output = self.server.model(torch.unsqueeze(frame, 0))
            anomaly_map = np.moveaxis(
                (np.clip(frame.cpu().detach().numpy() - output[0].cpu().detach().numpy(), 0, 1) * 255).astype(np.uint8),
                0, -1)
            anomaly_map = cv2.imencode('.png', anomaly_map)[1].dumps()
            self.wfile.write(pack_response(200, anomaly_map))
        else:
            pass
        self.wfile.flush()
        self.rfile.flush()
        logging.info('Response sent')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    frame_server = socketserver.ThreadingTCPServer(('0.0.0.0', 5555), FrameHandler)
    # model = Autoencoder((128, 128), 32, (128, 128),
    #                     convolutional=True, dropout_rate=0,
    #                     bottleneck_activation=None).to('cuda')
    # model = ResNet50ViT(img_dim=128, pretrained_resnet=False, blocks=4,
    #                     num_classes=3, dim_linear_block=64, dim=256).to('cuda')
    model = ObstacleAvoidance().to('cuda')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    frame_server.model = model
    frame_server.transform = transforms.ToTensor()
    try:
        logging.info("Started server...")
        frame_server.serve_forever()
    except KeyboardInterrupt:
        frame_server.shutdown()
