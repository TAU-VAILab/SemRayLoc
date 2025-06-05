import lightning.pytorch as pl
import torch.optim as optim
import torch.nn.functional as F

# Adjust the import path if needed.
from .semantic_net_maskformer_small import semantic_net

class semantic_net_pl_maskformer_small(pl.LightningModule):
    """
    Lightning wrapper for the semantic_net (predicting semantic classes).
    Regularization is applied via dropout (inside the network) and
    weight decay in the optimizer.
    """
    def __init__(
        self,
        num_classes,
        lr=1e-3,
        F_W=3/8,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.F_W = F_W
        self.num_classes = num_classes
        self.encoder = semantic_net(num_classes=num_classes)

    def configure_optimizers(self):
        # Use AdamW with weight decay for additional regularization.
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Extract inputs from the batch
        images = batch["ref_img"]
        masks = batch["ref_mask"] if "ref_mask" in batch else None
        target = batch["ref_semantics"].long()

        # Forward pass through the network
        logits, attn_2d, prob = self.encoder(images, masks)

        # Permute logits to shape [N, num_classes, fW] for cross-entropy loss
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, target)

        self.log("cross_entropy_loss-train", loss)
        self.log("loss-train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["ref_img"]
        masks = batch["ref_mask"] if "ref_mask" in batch else None
        target = batch["ref_semantics"].long()

        logits, attn_2d, prob = self.encoder(images, masks)
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, target)

        self.log("cross_entropy_loss-valid", loss)
        self.log("loss-valid", loss)
        return loss
