import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch import LightningModule

from best_k_net import best_k_net

class best_k_net_pl(LightningModule):
    """
    Lightning module wrapper for best_k_net.
    It computes cross-entropy loss against the ground-truth best index.
    
    The input batch is expected to have keys:
      - "ref_img": (N, 3, 360, 640)
      - "depth_vec": (N, 5, 40)
      - "sem_vec": (N, 5, 40)
      - "best_index": Tensor or int in [1,5] (1-indexed) -- target candidate.
    """
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Initialize the network.
        self.model = best_k_net()
        # Use CrossEntropyLoss which expects targets in the range [0, num_classes-1].
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image, depth_vec, sem_vec):
        """
        Forward pass.
        Args:
          image: (N, 3, 360, 640)
          depth_vec: (N, 5, 40)
          sem_vec: (N, 5, 40)
        Returns:
          logits: (N, 5)
        """
        return self.model(image, depth_vec, sem_vec)
    
    def training_step(self, batch, batch_idx):
        images = batch["ref_img"]
        depth_vec = batch["depth_vec"]
        sem_vec = batch["sem_vec"]
        # Convert best_index (1-indexed) to 0-indexed target.
        targets = batch["best_index"] - 1  # assuming best_index is a tensor
        logits = self(images, depth_vec, sem_vec)  # (N, 5)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["ref_img"]
        depth_vec = batch["depth_vec"]
        sem_vec = batch["sem_vec"]
        targets = batch["best_index"] - 1
        logits = self(images, depth_vec, sem_vec)
        loss = self.criterion(logits, targets)
        self.log("val_loss", loss)
        # Calculate accuracy.
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log("val_acc", acc)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# -----------------------------
# For Testing the Lightning Module Independently
# -----------------------------
if __name__ == '__main__':
    # Create dummy data.
    N = 2
    image = torch.randn(N, 3, 360, 640)
    depth_vec = torch.randn(N, 5, 40)
    sem_vec = torch.randn(N, 5, 40)
    # Assume best_index is given as 1-indexed tensor.
    best_index = torch.tensor([3, 5], dtype=torch.long)
    
    batch = {
        "ref_img": image,
        "depth_vec": depth_vec,
        "sem_vec": sem_vec,
        "best_index": best_index
    }
    
    model = best_k_net_pl(lr=1e-3)
    # Test forward
    logits = model(image, depth_vec, sem_vec)
    print("Logits shape:", logits.shape)
    # Test training step.
    loss = model.training_step(batch, 0)
    print("Training loss:", loss.item())
