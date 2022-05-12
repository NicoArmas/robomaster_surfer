import torch
import torch.nn as nn
import sklearn.metrics as metrics
from tqdm.auto import tqdm
from utils import compute_auc_score

import numpy as np
from torchinfo import summary

from .bottleneck import Bottleneck
from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder
from .decoder import Decoder
from .encoder import Encoder


class Autoencoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size=128,
                 convolutional=True, dropout_rate=0.5, config=None, device=None, bottleneck_activation=None):
        """
        The function takes in the input size, hidden size, output size, batch size, convolutional, dropout rate, config,
        device, and bottleneck activation.

        The function then sets the device to cuda if it's available, otherwise it sets it to cpu.

        The function then sets the hidden size, input size, and batch size.

        If the config is not None, then the function sets the input size, hidden size, output size, convolutional, and
        dropout rate.

        The function then sets the convolutional to convolutional.

        If convolutional is True, then the function sets the encoder to ConvolutionalEncoder, the bottleneck to Bottleneck,
        and the decoder to ConvolutionalDecoder.

        If convolutional is False, then the function sets the encoder to Encoder, and the decoder to Decoder.

        The

        :param input_size: The size of the input vector
        :param hidden_size: the size of the bottleneck layer
        :param output_size: The size of the output of the autoencoder
        :param batch_size: The number of samples per batch to load, defaults to 128 (optional)
        :param convolutional: whether or not to use convolutional layers, defaults to True (optional)
        :param dropout_rate: The dropout rate for the encoder and decoder
        :param config: a dictionary containing the parameters of the model
        :param device: the device to use for training
        :param bottleneck_activation: The activation function to use in the bottleneck layer
        """

        super(Autoencoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size

        if config is not None:
            input_size = config["input_size"]
            hidden_size = config["hidden_size"]
            output_size = config["output_size"]
            convolutional = config["convolutional"]
            dropout_rate = config["dropout_rate"]
        self.convolutional = convolutional
        if convolutional:
            self.encoder = ConvolutionalEncoder(dropout_rate=dropout_rate)
            if self.hidden_size > 0:
                self.bottleneck = Bottleneck((self.batch_size, 4, 4), hidden_size, bottleneck_activation)
            self.decoder = ConvolutionalDecoder(output_size, dropout_rate=dropout_rate)
        else:
            self.encoder = Encoder(input_size, hidden_size, output_size, device=self.device)
            self.decoder = Decoder(input_size, hidden_size, output_size, device=self.device)

        self.threshold = 0
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "convolutional": convolutional,
            "dropout_rate": dropout_rate
        }

    def summary(self, input_size):
        """
        It takes a model and an input size, and returns a string that summarizes the model's architecture

        :param input_size: The size of the input tensor
        :return: The summary function is returning the summary of the model.
        """
        return summary(self, input_size=input_size, device=self.device)

    def forward(self, x):
        """
        The encoder takes in an image, and outputs a vector of size (batch_size, hidden_size).

        The bottleneck takes in the output of the encoder, and outputs a vector of size (batch_size, 4*4*hidden_size).

        The decoder takes in the output of the bottleneck, and outputs a vector of size (batch_size, 784).

        The output of the decoder is the reconstructed image.

        The output of the encoder is the latent representation of the image.

        The output of the bottleneck is the latent representation of the image, but flattened.

        The bottleneck is only used if the hidden_size is greater than 0.

        The bottleneck is only used if the convolutional flag is set to True.

        The bottleneck is only used if the convolutional flag is set to True and the hidden_size is greater than

        :param x: the input to the model
        :return: The encoded and decoded outputs of the autoencoder.
        """
        encoder_out = self.encoder(x)
        if self.convolutional:
            if self.hidden_size > 0:
                bottleneck_out, encoded = self.bottleneck(encoder_out)
            else:
                bottleneck_out = torch.zeros_like(encoder_out.view(-1, int(np.prod((self.batch_size, 4, 4))))).to(self.device)
                encoded = torch.zeros(self.batch_size, 1).to(self.device)
            decoded = self.decoder(bottleneck_out)
        else:
            encoded = encoder_out
            decoded = self.decoder(encoder_out)
        return encoded, decoded

    def set_threshold(self, threshold_data_loader):
        """
        > We take the output of the model, and use it to calculate the ROC curve. We then take the second threshold from the
        ROC curve, and set it as the threshold for the model

        :param threshold_data_loader: a DataLoader object that contains the data to be used to set the threshold
        """
        self.eval()
        with torch.no_grad():
            for x, _ in tqdm(threshold_data_loader):
                x = x.to(self.device)
                encoded, y, _ = self(x)
                y_test = (x > 0.5).reshape(-1).cpu().detach().numpy().astype(np.uint8).tolist()
                y_score = y.reshape(-1).cpu().detach().numpy().astype(float).tolist()

                _, _, thresholds = metrics.roc_curve(y_test, y_score)
                self.threshold = thresholds[1]
