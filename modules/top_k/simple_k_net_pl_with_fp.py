import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule

from simple_k_net_with_fp import SimpleKNetWithFP

class simple_k_net_pl_with_fp(LightningModule):
    """
    PyTorch Lightning module for training a SimpleKNetWithFP model with only
    the first 3 candidates from each batch (originally K=5).

    If best_index is 4 or 5, we treat it as candidate #0 after truncation.
    """
    def __init__(self, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = SimpleKNetWithFP(k=3)  # Model expects exactly 3 candidates

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, image, depth_vec, sem_vec, semantic_map, k_positions):
        return self.model(image, depth_vec, sem_vec, semantic_map, k_positions)

    def training_step(self, batch, batch_idx):
        images       = batch["ref_img"]      # (N, 3, 360, 640)
        depth_vec    = batch["depth_vec"]    # (N, 5, 40)
        sem_vec      = batch["sem_vec"]      # (N, 5, 40)
        semantic_map = batch["semantic_map"] # (N, 1, 300, 300)
        k_positions  = batch["k_positions"]  # (N, 5, 3)
        best_index   = batch["best_index"]   # (N,) in {1..5}

        # Truncate the candidate dimension from 5 -> 3
        depth_vec   = depth_vec[:, :3, :]     # (N, 3, 40)
        sem_vec     = sem_vec[:, :3, :]       # (N, 3, 40)
        
        k_positions = [[list(p) for p in pos] for pos in k_positions]  
        k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=images.device)
        k_positions_tensor = k_positions_tensor[:3, :,  :]
        k_positions_tensor = k_positions_tensor.permute(1, 0, 2)  # Shape: [K, N, 3]

        mask = (best_index > 3)
        best_index[mask] = 1  # now 4 or 5 => 1

        # Convert from 1-based to 0-based indexing for CrossEntropy
        # So valid values are now {1,2,3} → {0,1,2}, and {4,5} → {1} → {0}.
        targets = best_index - 1  # shape (N,)

        logits = self(images, depth_vec, sem_vec, semantic_map, k_positions_tensor)  # (N, 3)
        loss = self.criterion(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images       = batch["ref_img"]
        depth_vec    = batch["depth_vec"]
        sem_vec      = batch["sem_vec"]
        semantic_map = batch["semantic_map"]
        k_positions  = batch["k_positions"]
        best_index   = batch["best_index"]

        # Truncate
        depth_vec   = depth_vec[:, :3, :]
        sem_vec     = sem_vec[:, :3, :]
        
        k_positions = [[list(p) for p in pos] for pos in k_positions]  
        k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=images.device)
        k_positions_tensor = k_positions_tensor[:3, :,  :]
        k_positions_tensor = k_positions_tensor.permute(1, 0, 2)  # Shape: [K, N, 3]

        # Reassign best_index if > 3
        mask = (best_index > 3)
        best_index[mask] = 1
        targets = best_index - 1

        logits = self(images, depth_vec, sem_vec, semantic_map, k_positions_tensor)
        loss = self.criterion(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Testing the Lightning Module
# -----------------------------
if __name__ == "__main__":
    # Example dummy batch with K=5
    N = 2
    k = 5

    image = torch.randn(N, 3, 360, 640)
    depth_vec = torch.randn(N, k, 40)
    sem_vec = torch.randn(N, k, 40)
    semantic_map = torch.randn(N, 1, 300, 300)
    k_positions = torch.randn(N, k, 3)

    # Suppose we have best_index in {2, 5}
    best_index = torch.tensor([2, 5], dtype=torch.long)
    batch = {
        "ref_img": image,
        "depth_vec": depth_vec,
        "sem_vec": sem_vec,
        "semantic_map": semantic_map,
        "k_positions": k_positions,
        "best_index": best_index
    }

    model_pl = simple_k_net_pl_with_fp(lr=1e-3, class_weights=[1.0, 1.0, 1.0])

    # Run training step
    loss = model_pl.training_step(batch, 0)
    print("Training step loss:", loss.item())
