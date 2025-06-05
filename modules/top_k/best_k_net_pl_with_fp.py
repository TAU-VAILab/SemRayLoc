import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule

from best_k_net_with_fp import best_k_net_with_fp

class best_k_net_pl_with_fp(LightningModule):
    """
    Lightning module wrapper for best_k_net_with_fp.
    It computes cross-entropy loss against the ground-truth best index.
    
    The input batch is expected to have keys:
      - "ref_img": (N, 3, 360, 640)
      - "depth_vec": (N, 5, 40)
      - "sem_vec": (N, 5, 40)
      - "semantic_map": (N, 1, 300, 300)
      - "k_positions": list of length N, each element is list of 5 (x, y, o)
      - "best_index": Tensor or int in [1,5] (1-indexed) -- target candidate.
    """
    def __init__(self, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Initialize the network.
        self.model = best_k_net_with_fp()
        # Use CrossEntropyLoss which expects targets in the range [0, num_classes-1].
        self.criterion = nn.CrossEntropyLoss()
        
        if class_weights is not None:
            # Ensure class_weights is on the correct device
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image, depth_vec, sem_vec, semantic_map, k_positions):
        """
        Forward pass.
        Args:
          image: (N, 3, 360, 640)
          depth_vec: (N, 5, 40)
          sem_vec: (N, 5, 40)
          semantic_map: (N, 1, 300, 300)
          k_positions: Tensor of shape (N, 5, 3)  # Each candidate's (x, y, o)
        Returns:
          logits: Tensor of shape (N, 5)
        """
        return self.model(image, depth_vec, sem_vec, semantic_map, k_positions)
    
    def training_step(self, batch, batch_idx):
        images = batch["ref_img"]              # (N, 3, 360, 640)
        depth_vec = batch["depth_vec"]         # (N, 5, 40)
        sem_vec = batch["sem_vec"]             # (N, 5, 40)
        semantic_map = batch["semantic_map"]   # (N, 1, 300, 300)
        k_positions = batch["k_positions"]     # list of length N, each element is list of 5 (x, y, o)
        # Convert best_index (1-indexed) to 0-indexed target.
        targets = batch["best_index"] - 1      # (N,), dtype=torch.long
        
        # Convert k_positions from list to tensor
        # Assuming k_positions is a list of N elements, each is a list of 5 [x, y, o]
        k_positions = [[list(p) for p in pos] for pos in k_positions]  
        k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=images.device)  # (N, 5, 3)
                
        logits = self(images, depth_vec, sem_vec, semantic_map, k_positions_tensor)  # (N, 5)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["ref_img"]              # (N, 3, 360, 640)
        depth_vec = batch["depth_vec"]         # (N, 5, 40)
        sem_vec = batch["sem_vec"]             # (N, 5, 40)
        semantic_map = batch["semantic_map"]   # (N, 1, 300, 300)
        k_positions = batch["k_positions"]     # list of length N, each element is list of 5 (x, y, o)
        targets = batch["best_index"] - 1      # (N,), dtype=torch.long
        
        # Convert k_positions from list to tensor
        k_positions = [[list(p) for p in pos] for pos in k_positions]  
        k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=images.device)  # (N, 5, 3)
        
        logits = self(images, depth_vec, sem_vec, semantic_map, k_positions_tensor)  # (N, 5)
        loss = self.criterion(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Calculate accuracy.
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    
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
    semantic_map = torch.randn(N, 1, 300, 300)
    k_positions = [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]],
        [[1.6, 1.7, 1.8], [1.9, 2.0, 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7], [2.8, 2.9, 3.0]]
    ]  # list of N=2 elements, each is list of 5 [x, y, o]
    # Assume best_index is given as 1-indexed tensor.
    best_index = torch.tensor([3, 5], dtype=torch.long)
    
    batch = {
        "ref_img": image,
        "depth_vec": depth_vec,
        "sem_vec": sem_vec,
        "semantic_map": semantic_map,
        "k_positions": k_positions,
        "best_index": best_index
    }
    
    model = best_k_net_pl_with_fp(lr=1e-3)
    # Test forward
    logits = model(image, depth_vec, sem_vec, semantic_map, k_positions_tensor := torch.tensor(k_positions, dtype=torch.float32))
    print("Logits shape:", logits.shape)  # Expected: (N, 5)
    # Test training step.
    loss = model.training_step(batch, 0)
    print("Training loss:", loss.item())
