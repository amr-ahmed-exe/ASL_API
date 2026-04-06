import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any


class SignLanguageCSVDataset(Dataset):
    """Dataset class for loading sign language data from CSV."""
    
    def __init__(self, csv_file: str) -> None:
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to the CSV file containing the dataset
        """
        self.data = pd.read_csv(csv_file)
        self.labels = self.data['label'].values
        self.features = self.data.drop('label', axis=1).values.astype(float)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for the transformer architecture."""
    
    def __init__(self, input_dim: int, num_heads: int = 4) -> None:
        """
        Initialize the multi-head attention module.
        
        Args:
            input_dim: Dimension of the input features
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of same shape as input
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SignLanguageAttentionModel(nn.Module):
    """Transformer-based model for sign language recognition."""
    
    def __init__(self, landmark_dim: int = 3, num_landmarks: int = 21, 
                 hidden_dim: int = 256, num_classes: int = 24) -> None:
        """
        Initialize the sign language attention model.
        
        Args:
            landmark_dim: Dimension of each landmark (x,y,z)
            num_landmarks: Number of landmarks per hand
            hidden_dim: Hidden dimension of the transformer
            num_classes: Number of sign language classes
        """
        super(SignLanguageAttentionModel, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Input projection
        self.input_proj = nn.Linear(landmark_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention1 = MultiHeadAttention(hidden_dim)
        self.attention2 = MultiHeadAttention(hidden_dim)
        
        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_landmarks * landmark_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to sequence
        x = x.view(-1, self.num_landmarks, 3)
        
        # Initial projection
        x = self.input_proj(x)
        
        # First attention block
        attn_out = self.attention1(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn1(x)
        x = self.norm2(x + ffn_out)
        
        # Second attention block
        attn_out = self.attention2(x)
        x = self.norm3(x + attn_out)
        ffn_out = self.ffn2(x)
        x = x + ffn_out
        
        # Global average pooling
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        return x


def train_model(csv_file: str, epochs: int = 100, batch_size: int = 64, 
                learning_rate: float = 0.0005) -> None:
    """
    Train the sign language recognition model.
    
    Args:
        csv_file: Path to the CSV file containing the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    dataset = SignLanguageCSVDataset(csv_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageAttentionModel(num_classes=24).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    # Add lists to store metrics
    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        
        # Store training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss /= total_val
        val_acc = correct_val / total_val
        
        # Store validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "sign_language_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    print("Training curves have been saved as 'training_curves.png'")
    print("Training complete. Best model saved as 'sign_language_model.pth'.")


if __name__ == "__main__":
    csv_file = "landmarks_dataset.csv"
    train_model(csv_file, epochs=100, batch_size=64, learning_rate=0.0005)