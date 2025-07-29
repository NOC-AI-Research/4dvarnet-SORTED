import glob
import os
import time
from math import prod
from random import randint

import hydra
import kornia.filters as kfilts
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import xarray as xr

# from ocean4dvarnet.data import XrConcatDataset
# from ocean4dvarnet.models import ConvLstmGradModel
from ocean4dvarnet.models import (
    BaseObsCost,
    BilinAEPriorCost,
    ConvLstmGradModel,
    GradSolver,
    Lit4dVarNet,
)
from ocean4dvarnet.utils import cosanneal_lr_adam, get_triang_time_wei
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        _val_rec_weight = kwargs.pop(
            "val_rec_weight",
            kwargs["rec_weight"],
        )
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "val_rec_weight",
            torch.from_numpy(_val_rec_weight),
            persistent=False,
        )

        self._n_rejected_batches = 0

    def get_rec_weight(self, phase):
        rec_weight = self.rec_weight
        if phase == "val":
            rec_weight = self.val_rec_weight
        return rec_weight

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        # print(loss)
        if loss is None:
            self._n_rejected_batches += 1
        return loss

    def on_train_epoch_end(self):
        self.log(
            "n_rejected_batches",
            self._n_rejected_batches,
            on_step=False,
            on_epoch=True,
        )

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            print("bother")
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log(
            f"{phase}_gloss",
            grad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return loss, out


class XrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inp_da,
        tgt_da,
        slice={
            "time": np.arange(
                np.datetime64("2018-01-01"),
                np.datetime64("2024-01-01"),
                np.timedelta64(1, "M"),
                dtype="datetime64[M]",
            )
        },
        patch_dims=None,
    ):
        self.inp_da = inp_da.sel(slice)
        self.tgt_da = tgt_da.sel(slice)
        if patch_dims is None:
            self.patch_dims = {}
            for dim in self.inp_da:
                self.patch_dims[dim] = 1
        else:
            self.patch_dims = patch_dims

        self.indices = []
        self.patches_per_dim = {}
        for dim in self.patch_dims:
            if self.inp_da[dim].shape[0] % self.patch_dims[dim] != 0:
                raise Exception(
                    f"Patch size must be a factor of shape. Dimension {dim}, got patch size {self.patch_dims[dim]} and data size {self.inp_da[dim].shape[0]}."
                )
            self.patches_per_dim[dim] = int(
                self.inp_da[dim].shape[0] / self.patch_dims[dim]
            )

        for i in np.ndindex(tuple(self.patches_per_dim.values())):
            self.indices.append(i)

    def __len__(self):
        size = 1
        for dim in self.inp_da.dims:
            size = size * int(self.inp_da[dim].shape[0] / self.patch_dims[dim])

        return size

    def __getitem__(self, index):
        i2 = self.indices[index]
        patches = {}
        i = 0
        for dim in self.inp_da.dims:
            patches[dim] = np.arange(
                i2[i] * self.patch_dims[dim],
                i2[i] * self.patch_dims[dim] + self.patch_dims[dim],
                1,
            )

            i = i + 1

        return self.inp_da.isel(patches).to_numpy(), self.tgt_da.isel(
            patches
        ).to_numpy()


class XrDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        inp_da,
        tgt_da,
        tpatch_dims,
        tslice,
        vslice,
        vpatch_dims=None,
        num_workers: int = 1,
        pin_memory: bool = True,
        batch_size: int = 4,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )
        self.inp_da = inp_da
        self.tgt_da = tgt_da
        self.tslice = tslice
        self.vslice = vslice
        self.tpatch_dims = tpatch_dims
        if vpatch_dims is None:
            self.vpatch_dims = tpatch_dims
        else:
            self.vpatch_dims = vpatch_dims

    def train_dataloader(self):
        """Load the training dataset."""
        dataloader = DataLoader(
            XrDataset(
                self.inp_da,
                self.tgt_da,
                self.tslice,
                self.tpatch_dims,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        """Load the training dataset."""
        dataloader = DataLoader(
            XrDataset(
                self.inp_da,
                self.tgt_da,
                self.vslice,
                self.vpatch_dims,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader


inp = xr.open_dataset("/home/joncon/SORTED/ARGO_gridded_20182024_ro.nc")["TEMPERATURE"]
tgt = xr.open_dataset("/home/joncon/SORTED/RG_ARGO_NA_20042024_ro.nc")["TEMPERATURE"]


model_checkpoint = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./",
    filename="best-unsplit20250725_no-sst",
)

trainer = Trainer(
    max_epochs=500,
    callbacks=[model_checkpoint],
    log_every_n_steps=12,
    accelerator="auto",
    precision=32,
    strategy="ddp_find_unused_parameters_true",
    # accelerator="tpu", devices=8
)

tds = XrDataset(
    inp_da=inp,
    tgt_da=tgt,
    patch_dims={"time": 12, "lat": 14, "lon": 38},
    slice={
        "time": np.arange(
            np.datetime64("2018-01-01"),
            np.datetime64("2024-01-01"),
            np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    },
)
tdl = DataLoader(
    dataset=XrDataset(
        inp_da=inp,
        tgt_da=tgt,
        patch_dims={"time": 12, "lat": 14, "lon": 38},
        slice={
            "time": np.arange(
                np.datetime64("2018-01-01"),
                np.datetime64("2024-01-01"),
                np.timedelta64(1, "M"),
                dtype="datetime64[M]",
            )
        },
    )
)

inp_da = xr.ones_like(inp)
tgt_da = xr.ones_like(tgt)
print(tgt.dtype)

# rng = np.random.default_rng()
# inp_da.values = rng.random(size=inp_da.shape, dtype="float32")
# tgt_da.values = rng.random(size=tgt_da.shape, dtype="float32")


datamodule = XrDataModule(
    inp_da=inp_da,
    tgt_da=tgt_da,
    tpatch_dims={"time": 12, "lat": 14, "lon": 38},
    tslice={
        "time": np.arange(
            np.datetime64("2018-01-01"),
            np.datetime64("2024-01-01"),
            np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    },
    vslice={
        "time": np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2025-01-01"),
            np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    },
    num_workers=0,
)
# with hydra.initialize("4dvarnet-global-mapping/config/xp/"):
#    cfg = hydra.compose("sorted")
# model = hydra.utils.call(cfg.model)


class Encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.Prior_Cost = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x):
        return self.Prior_Cost(x)


class ConvLSTMGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLSTM = nn.Sequential(
            nn.Conv2d(12, 268, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(268, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Identity(),
        )

    def forward(self, x):
        return self.ConvLSTM(x)


class FourDVarNet(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        # x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = FourDVarNet(Encode(), ConvLSTMGradModel())


es = L.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", mode="min", patience=3
)

trainer = L.Trainer(
    default_root_dir="~/SORTED/4dvarnet/ocean4dvarnet/logging/"
    + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    + "/",
    callbacks=[es],
    max_epochs=100,
)
trainer.fit(model=model, datamodule=datamodule)
