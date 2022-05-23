import math
from datetime import datetime

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

import wandb
from host.autoencoder import Autoencoder
from host.utils import compute_auc_score, plot_stages
from dataset import LaneDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# WANDB_PROJECT = ""
# USE_WANDB = True
DEBUG = False


class Trainer:

    def __init__(self, autoencoder, train_loader, val_loader, test_loader, epochs=100, lr=1e-3,
                 loss_fn=nn.MSELoss(), optimizer=optim.Adam, run_number=0, denoise=False, noise_factor=0, split=0,
                 batch_size=64, samples_per_epoch=None, run_name=None, wandb_cfg=None, use_wandb=False):
        """
        This function initializes the class with the following parameters:

        - autoencoder: the autoencoder model
        - train_loader: the training data loader
        - val_loader: the validation data loader
        - test_loader: the test data loader
        - epochs: the number of epochs to train for
        - lr: the learning rate
        - loss_fn: the loss function to use
        - optimizer: the optimizer to use
        - run_number: the run number
        - denoise: whether to use denoising
        - noise_factor: the noise factor to use
        - split: the split of the data to use
        - batch_size: the batch size to use
        - samples_per_epoch: the number of samples per epoch
        - run_name: the name of the run
        - wandb_cfg: the wandb configuration
        - use_wandb: whether to

        :param autoencoder: The autoencoder model to train
        :param train_loader: A PyTorch DataLoader object for the training set
        :param val_loader: A DataLoader object for the validation set
        :param test_loader: the test set
        :param epochs: Number of epochs to train for
        :param lr: learning rate
        :param loss_fn: The loss function to use. Defaults to MSELoss
        :param optimizer: The optimizer to use
        :param run_number: This is the number of the run. This is used to create a unique name for the run, defaults to 0
        (optional)
        :param denoise: Whether to use denoising autoencoder or not, defaults to False (optional)
        :param noise_factor: The amount of noise to add to the input images, defaults to 0 (optional)
        :param split: the percentage of the dataset to use for training. The rest is used for validation, defaults to 0
        (optional)
        :param batch_size: The number of samples to use in each batch, defaults to 64 (optional)
        :param samples_per_epoch: The number of samples to use per epoch. If None, all samples will be used
        :param run_name: The name of the run. This will be used to create a folder to store the model and the results
        :param wandb_cfg: a dictionary of configuration parameters for wandb
        :param use_wandb: Whether to use wandb to log results, defaults to False (optional)
        """

        self.model = autoencoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.run_number = run_number
        self.denoise = denoise
        self.noise_factor = noise_factor
        self.split = split
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else len(train_loader.dataset)
        cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = run_name if run_name is not None else f'{cur_time}_run_{run_number}'
        self.use_wandb = use_wandb
        self.wandb_cfg = wandb_cfg

    def fit(self, test=True, test_path=None, overwrite=False):
        """
        It trains the model, and if test is True, it tests the model on the test_path

        :param test: whether to run the test set, defaults to True (optional)
        :param test_path: the path to the test data
        :param overwrite: If True, will overwrite the model if it already exists, defaults to False (optional)
        :return: The model trained and tested.
        """

        if os.path.exists(f"{self.run_name}.pt") and not overwrite:
            print(f"Model {self.run_name} already exists. Use overwrite=True to overwrite")
            return

        if test and test_path is None:
            raise ValueError("test_path must be specified if test is True")

        if self.use_wandb:
            wandb.init(**self.wandb_cfg)
            wandb.watch(self.model, log="all")
            model_artifact = wandb.Artifact(f'{self.run_name}', type='model')
        if self.run_name is None:
            self.run_name = "run_" + str(self.run_number)
        if self.use_wandb:
            wandb.config.update({"run_name": self.run_name})
        self.train()
        torch.save(self.model.state_dict(), f"{self.run_name}.pt")
        if test:
            self.test(test_path)
        if self.use_wandb:
            model_artifact.add_file(f"{self.run_name}.pt")
            wandb.log_artifact(model_artifact)
            wandb.finish()

    def train(self):
        """
        > The function trains the autoencoder model using the training data, and then evaluates the model on the validation
        data
        """
        autoencoder = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        epochs = self.epochs
        lr = self.lr
        run_name = self.run_name
        split = self.split
        samples_per_epoch = self.samples_per_epoch
        denoise = self.denoise
        noise_factor = self.noise_factor
        train_loader = self.train_loader
        val_loader = self.val_loader
        run_number = self.run_number
        batch_size = self.batch_size

        optimizer = optimizer(autoencoder.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        criterion = loss_fn
        v_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        wandb.config = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "run_number": run_number,
            "split": split,
        }

        with tqdm(range(epochs), total=epochs, unit='epoch') as tepoch:
            for epoch in tepoch:
                mse_loss = 0
                mae_loss = 0
                losses = []
                thresh_losses = []
                samples = 0

                with tqdm(train_loader, total=len(train_loader), unit='batch', leave=True) as tbatch:
                    for data, _ in tbatch:
                        samples += data.size(0)
                        autoencoder.train()
                        data = data.to(DEVICE)
                        optimizer.zero_grad()
                        if denoise:
                            data_noisy = data
                            encoded, decoded = autoencoder(data_noisy)
                        else:
                            encoded, decoded = autoencoder(data)
                        loss = criterion(decoded, data)
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        mse_loss += loss.item()

                        with torch.no_grad():
                            mae_l = mae_criterion(decoded, data).item()
                            mae_loss += mae_l

                        if self.use_wandb:
                            wandb.log({
                                "step": samples / batch_size + epoch * samples_per_epoch / batch_size,
                                "epoch": epoch,
                                "Training MSE Loss": loss.item(),
                                "Training MAE Loss": mae_l})

                        tepoch.set_postfix(
                            samples=f'{samples}/{samples_per_epoch}',
                            MSE_Loss=mse_loss / ((samples / batch_size) + 1),
                            MAE_Loss=mae_loss / ((samples / batch_size) + 1),
                            batch=f'{(samples // batch_size)}/{int(math.ceil(samples_per_epoch / batch_size))}')

                        autoencoder.eval()
                        with torch.no_grad():
                            thresh_losses.append(v_criterion(decoded, data).item())

                if epoch % 1 == 0:
                    autoencoder.eval()
                    with torch.no_grad():
                        v_mse = []
                        v_mae = []
                        autoencoder.eval()
                        with torch.no_grad():
                            for j, (data, _) in enumerate(val_loader):
                                data = data.to(DEVICE)
                                encoded, decoded = autoencoder(data)

                                v_mse.append(criterion(decoded, data).item())
                                v_mae.append(mae_criterion(decoded, data).item())

                                if self.use_wandb:
                                    wandb.log({
                                        "epoch": epoch,
                                        "Validation MSE Loss": v_mse[-1],
                                        "Validation MAE Loss": v_mae[-1]})

                                tepoch.set_postfix(validation=True,
                                                   Validation_MSE=np.mean(v_mse),
                                                   Validation_MAE=np.mean(v_mae))

                    scheduler.step(np.mean(v_mse))

            with torch.no_grad():
                autoencoder.threshold = np.mean(thresh_losses) + np.std(thresh_losses)

    def test(self, path):
        """
        It takes a model, a test loader, and a path to save the images to, and it computes the AUC score for the model on
        the test set, and saves some images to the path

        :param path: the path to save the images to
        :return: The mean of the AUC score
        """
        autoencoder = self.model
        test_loader = self.test_loader

        autoencoder.eval()
        auc = []
        flags = {
            "empty_center": 3,
            "empty_left": 3,
            "empty_right": 3,
            "obstacles": 60,
        }
        with torch.no_grad():
            for i, (data, label) in enumerate(tqdm(test_loader)):
                data = data.to(DEVICE)
                encoded, decoded = autoencoder(data)

                aucloss, threshold = compute_auc_score(decoded, data, return_threshold=True)
                auc.append(aucloss)

                if np.any(flags):
                    for d, cls in zip(data, label):
                        if flags[cls]:
                            plot_stages(autoencoder, d, save=True, plot=False, path=path.format(cls))
                            if self.use_wandb:
                                wandb.log({'image': wandb.Image(path.format(cls))})
                            flags[cls] -= 1

        print(f'\ntest set AUC Score: {np.mean(auc)}')
        if self.use_wandb:
            wandb.log({"Test AUC": np.mean(auc)})
        return np.mean(auc)


def main():
    batch_size = 32
    epochs = 200
    lr = 1e-4
    dropout = 0
    train_set = LaneDataset('robomaster_surfer/vision/data/preprocessed')
    test_set = LaneDataset('robomaster_surfer/vision/data/preprocessed_test')
    train_set, valid_set = torch.utils.data.random_split(train_set, [int(len(train_set) * 0.95),
                                                                     int(len(train_set) * 0.05)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4,
                                               pin_memory=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=4,
                                               pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=4,
                                              pin_memory=True, shuffle=False)

    input_size = output_size = (128, 128)
    for hidden_size in [16]:

        model = Autoencoder(input_size, hidden_size, output_size,
                            convolutional=True, dropout_rate=dropout,
                            bottleneck_activation=None).to(DEVICE)

        wandb_cfg = {
            "project": "robomaster_surfer_test_model",
            "entity": "axhyra",
            "name": "autoencoder",
            "group": f'bottleneck:{hidden_size}',
        }

        trainer_cfg = {
            "autoencoder": model,
            "train_loader": train_loader,
            "val_loader": valid_loader,
            "test_loader": test_loader,
            "epochs": epochs,
            "lr": lr,
            "loss_fn": nn.MSELoss(),
            "optimizer": optim.Adam,
            "run_number": 0,
            "denoise": False,
            "batch_size": batch_size,
            "use_wandb": True,
            "wandb_cfg": wandb_cfg,
            "run_name": f'bottleneck_{hidden_size}',
        }

        trainer = Trainer(**trainer_cfg)

        trainer.fit(test_path='./plots/{}.png')


if __name__ == "__main__":
    main()
