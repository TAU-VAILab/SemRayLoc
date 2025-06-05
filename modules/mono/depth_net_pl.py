# import lightning.pytorch as pl
from lightning.pytorch import LightningModule
import torch.optim as optim

from .depth_net import *


class depth_net_pl(LightningModule):
    """
    lightning wrapper for the depth_net
    """

    def __init__(
        self,
        shape_loss_weight=None,
        lr=1e-3,
        d_min=0.1,
        d_max=15.0,
        d_hyp=-0.2,
        D=128,
        F_W=3 / 8,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.F_W = F_W
        self.encoder = depth_net(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Use 'ref_img' instead of 'img'
        rays, attn_2d, _ = self.encoder(
            batch["ref_img"], batch["ref_mask"] if "ref_mask" in batch else None
        )
        loss = F.l1_loss(rays, batch["ref_depth"])
        self.log("l1_loss-train", loss)

        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(rays, batch["ref_depth"]).mean()
            )
            loss += shape_loss
            self.log("shape_loss-train", shape_loss)

        self.log("loss-train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Use 'ref_img' instead of 'img'
        rays, attn_2d, prob = self.encoder(
            batch["ref_img"], batch["ref_mask"] if "ref_mask" in batch else None
        )
        loss = F.l1_loss(rays, batch["ref_depth"])
        self.log("l1_loss-valid", loss)
        if self.shape_loss_weight is not None:
            shape_loss = 1 - F.cosine_similarity(rays, batch["ref_depth"]).mean()
            loss += shape_loss
            self.log("shape_loss-valid", shape_loss)

        self.log("loss-valid", loss)

