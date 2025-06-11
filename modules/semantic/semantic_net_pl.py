import lightning.pytorch as pl
import torch.optim as optim
import torch.nn.functional as F
from .semantic_and_room_type_with_cls import semantic_net
class semantic_net_pl(pl.LightningModule):
    """
    Lightning wrapper for a multi-task net that predicts:
      1) 40 semantic rays
      2) 1 global room-type
    """
    def __init__(
        self,
        num_ray_classes = 4,           # e.g., the # of semantic categories for each ray
        num_room_types=16,         # len(VALID_ROOM_TYPES)
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.num_ray_classes = num_ray_classes
        self.num_room_types = num_room_types

        self.model = semantic_net(
            num_ray_classes=self.num_ray_classes,
            num_room_types=self.num_room_types
        )

            
    def forward(self, x, mask = None):
        return self.model(x, mask)

    def configure_optimizers(self):
        # Typical AdamW + some weight decay
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        We expect batch to have:
          batch["ref_img"]: shape (N,3,H,W)
          batch["ref_mask"]: (optional) shape (N,H,W)
          batch["ref_semantics"]: shape (N,40) for ray labels
          batch["room_label"]: shape (N,) in [0..num_room_types-1]
        """
        images = batch["ref_img"]
        mask = batch.get("ref_mask", None)

        ray_labels = batch["ref_semantics"].long()  # (N,40)
        room_labels = batch["room_label"].long()    # (N,)

        # Forward pass
        ray_logits, room_logits, attn = self.model(images, mask=mask)
        # ray_logits: (N,40,num_ray_classes)
        # room_logits: (N,num_room_types)

        # Ray classification: cross-entropy expects shape (N, num_ray_classes, 40)
        ray_logits_perm = ray_logits.permute(0, 2, 1)
        loss_rays = F.cross_entropy(ray_logits_perm, ray_labels)

        # Room classification: cross-entropy => shape (N,num_room_types), (N,)
        loss_room = F.cross_entropy(room_logits, room_labels)

        loss = loss_rays + loss_room
        self.log("loss_rays_train", loss_rays, prog_bar=True)
        self.log("loss_room_train", loss_room, prog_bar=True)
        self.log("loss_train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["ref_img"]
        mask = batch.get("ref_mask", None)

        ray_labels = batch["ref_semantics"].long()  # (N,40)
        room_labels = batch["room_label"].long()    # (N,)
        
        # Forward pass
        ray_logits, room_logits, attn = self.model(images, mask=mask)
        ray_logits_perm = ray_logits.permute(0, 2, 1)

        loss_rays = F.cross_entropy(ray_logits_perm, ray_labels)
        loss_room = F.cross_entropy(room_logits, room_labels)
        loss = loss_rays + loss_room

        # ----------------------------------------------------
        # Compute accuracies
        # ----------------------------------------------------
        # 1) Rays: ray_logits_perm shape => (N, num_ray_classes, 40)
        #    We want argmax over channel dim => (N,40)
        pred_rays = ray_logits_perm.argmax(dim=1)  # shape (N,40)
        correct_rays = (pred_rays == ray_labels).float()  # elementwise
        acc_rays = correct_rays.mean()  # average across all rays and batch => scalar

        # 2) Room type: room_logits => (N,num_room_types)
        pred_room = room_logits.argmax(dim=1)   # (N,)
        acc_room = (pred_room == room_labels).float().mean()

        # ----------------------------------------------------
        # Log everything
        # ----------------------------------------------------
        self.log("cross_entropy_loss-valid", loss)
        self.log("loss-valid", loss)
        
        self.log("loss_rays_val", loss_rays)
        self.log("loss_room_val", loss_room)
        

        self.log("acc_rays_val", acc_rays, prog_bar=True)
        self.log("acc_room_val", acc_room, prog_bar=True)

        return {
            "loss": loss,
            "acc_rays": acc_rays,
            "acc_room": acc_room
        }
