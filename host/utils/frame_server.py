import pickle
import socketserver
from typing import Callable, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from host.autoencoder import Autoencoder

HEADERSIZE = 10


class FrameServer(socketserver.BaseServer):
    def __init__(self,
                 server_address: Tuple[str, int],
                 RequestHandlerClass: Callable[..., socketserver.StreamRequestHandler], model):
        super().__init__(server_address, RequestHandlerClass)


def pack_response(code: int, data: bytes):
    res = bytes(f'{code:<{HEADERSIZE}}', 'utf-8') + \
          bytes(f'{len(data):<{HEADERSIZE}}', 'utf-8') + \
          data
    print(f'Sending response, return code: {res[:10].split()[0]}, length: {res[11:20].split()[0]}')
    return res


class FrameHandler(socketserver.StreamRequestHandler):

    def send_error(self, code: int = 404):
        self.wfile.write(pack_response(code, b'Error'))

    def handle(self):
        print(f'Got connection from {self.client_address}')
        msg = self.rfile.readline().split()
        print(msg)
        req = msg[0].decode('utf-8')
        content_size = int(msg[1].decode('utf-8'))
        if not req:
            return
        content = self.rfile.read(content_size)
        frame = cv2.imdecode(pickle.loads(content), cv2.IMREAD_COLOR)
        cv2.imwrite('frame.png', frame)
        frame = self.server.transform(frame).to('cuda')
        print(frame.shape)

        if req == 'get_movement':
            response = np.array([0 if np.random.random() <= 0.5 else 1 for _ in range(9)]).dumps()
            self.wfile.write(pack_response(200, response))
        elif req == 'get_anomaly_map':
            self.server.model.eval()
            _, output = self.server.model(torch.unsqueeze(frame, 0))
            anomaly_map = np.clip(frame[0].cpu().detach().numpy() - output[0][0].cpu().detach().numpy(), 0, 1)
            anomaly_map = np.moveaxis([anomaly_map] * 3, 0, -1)
            anomaly_map = (anomaly_map * 255).astype('uint8')
            anomaly_map = cv2.imencode('.png', anomaly_map)[1].dumps()
            self.wfile.write(pack_response(200, anomaly_map))
        else:
            pass
        self.wfile.flush()
        self.rfile.flush()
        print('Response sent')


if __name__ == '__main__':
    frame_server = socketserver.ThreadingTCPServer(('0.0.0.0', 5555), FrameHandler)
    model = Autoencoder((128, 128), 16, (128, 128),
                        convolutional=True, dropout_rate=0,
                        bottleneck_activation=None).to('cuda')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    frame_server.model = model
    frame_server.transform = transforms.ToTensor()
    try:
        frame_server.serve_forever()
    except KeyboardInterrupt:
        frame_server.shutdown()
