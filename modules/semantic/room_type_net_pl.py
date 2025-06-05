import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from modules.semantic.room_type_net import room_type_net

class room_type_net_pl(pl.LightningModule):
    """
    Lightning wrapper for a net that predicts room type.
    """
    def __init__(
        self,
        num_room_types=16,  # e.g., number of valid room types
        lr=1e-3,
        embed_dim=64
    ) -> None:
        super().__init__()
        self.lr = lr
        self.num_room_types = num_room_types

        # The actual model
        self.model = room_type_net(
            num_room_types=self.num_room_types,
            embed_dim=embed_dim
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Expects batch to have:
          batch["ref_img"]: shape (N,3,H,W)
          batch["ref_mask"]: (optional) shape (N,H,W)
          batch["room_label"]: shape (N,) in [0, num_room_types-1]
        """
        images = batch["ref_img"]
        mask = batch.get("ref_mask", None)
        room_labels = batch["room_label"].long()

        room_logits, attn = self.model(images, mask=mask)
        loss_room = F.cross_entropy(room_logits, room_labels)
        self.log("loss_train", loss_room, prog_bar=True)
        return loss_room

    def validation_step(self, batch, batch_idx):
        images = batch["ref_img"]
        mask = batch.get("ref_mask", None)
        room_labels = batch["room_label"].long()

        room_logits, attn = self.model(images, mask=mask)
        loss_room = F.cross_entropy(room_logits, room_labels)

        pred_room = room_logits.argmax(dim=1)
        acc_room = (pred_room == room_labels).float().mean()

        self.log("loss-valid", loss_room)
        self.log("acc_room_val", acc_room, prog_bar=True)

        return {"loss": loss_room, "acc_room": acc_room}
