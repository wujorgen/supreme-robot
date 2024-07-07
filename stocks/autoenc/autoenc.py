import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch import nn

from data_modules import CallDataModule

##### DEFINE A BASIC FFNN #####

##### DEFINE AUTOENCODER MODEL #####
"""
https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
"""


##### DEFINE FEEDFORWARDS MDOEL #####
class PLNetwork(pl.LightningModule):
    """
    Outline of Lightning Module: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(self):
        super(PLNetwork, self).__init__()
        self.loss_fun = F.mse_loss
        self.layers = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fun(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fun(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


##### TRAIN THE MODEL #####
if __name__ == "__main__":
    call_data = CallDataModule()
    # for idx, (X, y) in enumerate(call_data.train_dataloader()):
    #    print(idx, y)
    #    breakpoint()
    torch.set_float32_matmul_precision("high")
    logger = CSVLogger("logs", name="my_exp_name")
    model = PLNetwork()

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        max_epochs=100,
        enable_progress_bar=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )

    trainer.fit(model=model, datamodule=call_data)
    breakpoint()
