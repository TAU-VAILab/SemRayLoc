import lightning.pytorch as pl
import torch.optim as optim
import torch.nn.functional as F

from .semantic_net import *


class semantic_net_pl(pl.LightningModule):
    """
    lightning wrapper for the semantic_net (now predicting semantic classes)
    """

    def __init__(
        self,
        num_classes,
        shape_loss_weight=None,
        lr=1e-3,    
        F_W=3 / 8,
    ) -> None:
        super().__init__()
        self.lr = lr    
        self.F_W = F_W
        self.num_classes = num_classes
        self.encoder = semantic_net(
            num_classes=num_classes,        
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # train the semantic predictions
        logits, attn_2d, _ = self.encoder(
            batch["ref_img"], batch["ref_mask"] if "ref_mask" in batch else None
        )
        
        logits = logits.permute(0, 2, 1)  # Now logits is [16, 3, 40]
        target = batch["ref_semantics"].long()
        loss = F.cross_entropy(logits, target)
        
        self.log("cross_entropy_loss-train", loss)

        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(logits, target).mean()
            )
            loss += shape_loss
            self.log("shape_loss-train", shape_loss) # TODO decide if to add

        self.log("loss-train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validate the semantic predictions
        logits, attn_2d, prob = self.encoder(
            batch["ref_img"], batch["ref_mask"] if "ref_mask" in batch else None
        )

        # Permute logits to [16, 3, 40] for cross_entropy
        logits = logits.permute(0, 2, 1)

        # Ensure target is of type LongTensor
        target = batch["ref_semantics"].long()

        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, target)

        self.log("cross_entropy_loss-valid", loss)

        if self.shape_loss_weight is not None:
            # Assuming cosine similarity computation is still valid for logits and target
            shape_loss = 1 - F.cosine_similarity(logits, target).mean() # TODO decide if to add
            loss += shape_loss
            self.log("shape_loss-valid", shape_loss)

        self.log("loss-valid", loss)
        return loss
