import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

# IMPORTANT: Make sure this import points to the ResNet-based network file.
# from modules.semantic.room_type_pred.room_type_pred_resnet50_net import room_type_resnet50_net
from modules.semantic.room_type_pred.room_type_pred_resnet50_pretrained_net import room_type_resnet50_net

class room_type_pred_resnet50_pl(pl.LightningModule):
    """
    Lightning module for a classification network with attention,
    using a ResNet-50 backbone.

    Expects a batch dict with:
      "ref_img":   (N,3,H,W) input images
      "room_label": (N,)      integer class labels
    """
    def __init__(self, num_classes=16, lr=1e-3, embed_dim=256):
        super().__init__()
        self.lr = lr
        self.model = room_type_resnet50_net(num_classes=num_classes, embed_dim=embed_dim)
    
    def forward(self, x):
        """
        Forward pass: returns (logits, attn)
        """
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images = batch["ref_img"]
        labels = batch["room_label"].long()
        logits, attn = self.model(images)
        loss = F.cross_entropy(logits, labels)
        self.log("loss_train", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["ref_img"]
        labels = batch["room_label"].long()
        logits, attn = self.model(images)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("loss-valid", loss)
        self.log("acc_room_val", acc, prog_bar=True)
        return {"loss": loss, "acc_room": acc}
