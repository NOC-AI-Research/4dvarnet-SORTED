import time

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import xarray as xr
from pytorch_lightning import (
    LightningDataModule,
    Trainer,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

#ds_path = "/noc/users/joncon/SORTED/4dvarnet-SORTED/"
# ds_path = "/home/joncon/SORTED/"
ds_path = "/gws/nopw/j04/aria_sorted/jdconey/"

log_path = (
#    "/noc/users/joncon/SORTED/4dvarnet-SORTED/logging/"
     "/gws/nopw/j04/aria_sorted/jdconey/code/SORTED/ocean4dvarnet/logging/"
    + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    + "/"
)
# log_path = (
#    "~/SORTED/4dvarnet/ocean4dvarnet/logging/"
#    + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
#    + "/"
# )


class XrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inp_da,
        tgt_da,
        slice={
            "time_counter": np.arange(
                np.datetime64("2018-01-01"),
                np.datetime64("2024-01-01"),
                np.timedelta64(1, "M"),
                dtype="datetime64[M]",
            )
        },
        patch_dims=None,
        transform=None,
    ):
        self.inp_da = inp_da.sel(slice,method='nearest',tolerance=np.timedelta64(3,"D"))
        self.tgt_da = tgt_da.sel(slice,method='nearest',tolerance=np.timedelta64(3,"D"))
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
        self.transform = transform

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
        if self.transform:
            inp_t = self.transform(self.inp_da.isel(patches).to_numpy())
        else:
            inp_t = self.inp_da.isel(patches).to_numpy()

        a = np.zeros_like(inp_t.flatten(), dtype=int)

        prob = np.random.random_sample()
        prob_val = int(prob * len(a))
        a[:prob_val] = 1
        np.random.shuffle(a)
        # a = a.astype(bool)
        a = a.reshape(inp_t.shape)
        inp_t = inp_t * a
        inp_t = np.where(np.isfinite(inp_t), inp_t, 0.0).astype(np.float32)


        return inp_t, self.tgt_da.isel(patches).to_numpy()


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
        transforms=None,
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
        self.transforms = transforms

    def train_dataloader(self):
        """Load the training dataset."""
        self.transforms = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.0], std=[1.0])]
        )
        dataloader = DataLoader(
            XrDataset(
                self.inp_da, self.tgt_da, self.tslice, self.tpatch_dims, self.transforms
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        """Load the training dataset."""
        self.transforms = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.0], std=[1.0])]
        )
        dataloader = DataLoader(
            XrDataset(
                self.inp_da, self.tgt_da, self.vslice, self.vpatch_dims, self.transforms
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader


#inp = xr.open_dataset(ds_path + "ARGO_gridded_20182024_3D.nc")["TEMPERATURE"]
#tgt = xr.open_dataset(ds_path + "RG_ARGO_NA_20042024_3D.nc")["TEMPERATURE"]
inp = xr.open_dataarray(ds_path + "NEMO_thetao_1976-2024.nc")[:,:72,:212,:306]
tgt = xr.open_dataarray(ds_path + "NEMO_thetao_1976-2024.nc")[:,:72,:212,:306]

ts = []
for val in inp["time_counter"]:
    val = val.values
    ts.append(val - np.timedelta64(14,'D'))
inp['time_counter'] = np.array(ts)

ts = []
for val in tgt["time_counter"]:
    val = val.values
    ts.append(val - np.timedelta64(14,'D'))
tgt['time_counter'] = np.array(ts)

print(tgt['time_counter'])

datamodule = XrDataModule(
    inp_da=inp,
    tgt_da=tgt,
    tpatch_dims={"time_counter": 6, "deptht": 18, "y": 106, "x": 102},
    tslice={
        "time_counter": np.arange(
            np.datetime64("1976-01-01"),
            np.datetime64("2013-01-01"),
            np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    },
    vslice={
        "time_counter": np.arange(
            np.datetime64("2013-01-01"),
            np.datetime64("2019-01-01"),
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
            nn.Conv3d(6, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512, 6, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )

    def forward(self, x):
        return self.Prior_Cost(x)


class ConvGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv3d(6, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.Conv3d(256, 1024, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(1024),
            nn.Conv3d(1024, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.Conv3d(256, 6, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Identity(),
        )

    def forward(self, x):
        return self.ConvNet(x)


class FlatLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.LSTM_mod = nn.Sequential(
            nn.Conv3d(6, 6, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(6),
            nn.MaxPool3d(kernel_size=(3,5,5),stride=(3,5,5),padding=(0,0,0),dilation=(1,2,2)),
            nn.Flatten(2, 4),
            nn.LSTM(1600, 1600),
        )

    def forward(self, x):
        seq_out = self.LSTM_mod(x)[0]
#        print(seq_out.shape)
        batch_size = seq_out.shape[0]

        oot = torch.reshape(seq_out, (batch_size, 6, 4, 20, 20))
        oot2 = nn.ConvTranspose3d(in_channels=6, out_channels=6, 
                kernel_size=(5,9,9), stride=(3,3,3), padding=(0,0,2),dilation=(2,6,6))
        return oot2(oot)


class FourDVarNet(L.LightningModule):
    def __init__(self, encoder, flat_lstm, decoder):
        super().__init__()
        self.automatic_optimization = True
        self.encoder = encoder
        self.flat_lstm = flat_lstm
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        z = self.encoder(x)
        z = self.flat_lstm(z)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
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
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        z = self.encoder(x)
        z = self.flat_lstm(z)
        x_hat = self.decoder(z)
        return x_hat


model = FourDVarNet(Encode(), FlatLSTM(), ConvGradModel())


es = L.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", mode="min", patience=200
)

ckpt_cb = L.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath=log_path,
    filename="4dvarnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
)

trainer = Trainer(
    default_root_dir=log_path,
    callbacks=[es, ckpt_cb],
    max_epochs=10000,
    log_every_n_steps=6,
    accumulate_grad_batches=2,
)
trainer.fit(model=model, datamodule=datamodule)
